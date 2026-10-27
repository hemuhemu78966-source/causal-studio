"""Regression tests for causal semantics, preprocessing, API and evaluation."""
import numpy as np
import pandas as pd
import networkx as nx
import pytest
from pgmpy.factors.discrete import TabularCPD
from src.models.bayesian_network import CausalBayesianNetwork
from src.inference.intervention import InterventionAnalysis
from src.recourse.recourse_optimizer import AlgorithmicRecourse
from src.discovery.structure_learning import CausalDiscovery
from backend.service import Preprocessor, StudioService
from backend.main import app
from fastapi.testclient import TestClient

def chain(labels=False):
    n=CausalBayesianNetwork();n.add_nodes(['X','M','Y']);n.add_edges([('X','M'),('M','Y')])
    states=['off','on'] if labels else [0,1]
    n.set_cpd('X',TabularCPD('X',2,[[.5],[.5]],state_names={'X':states}))
    n.set_cpd('M',TabularCPD('M',2,[[.9,.1],[.1,.9]],['X'],[2],state_names={'M':states,'X':states}))
    n.set_cpd('Y',TabularCPD('Y',2,[[.95,.05],[.05,.95]],['M'],[2],state_names={'Y':states,'M':states}))
    n.build_model();return n

def test_text_intervention_preserves_states():
    e=InterventionAnalysis(chain(True))
    assert e.do_intervention({'X':'on'},'Y')['on']==pytest.approx(.86)
    assert e.do_intervention({'X':'off'},'Y')['on']==pytest.approx(.14)

@pytest.mark.parametrize('invalid',[-1,2,'unknown'])
def test_invalid_intervention_rejected(invalid):
    with pytest.raises(ValueError): InterventionAnalysis(chain()).do_intervention({'X':invalid},'Y')

def test_target_intervention_is_point_mass():
    assert InterventionAnalysis(chain()).do_intervention({'Y':1},'Y')=={'0':0.,'1':1.}

def test_action_propagates_through_mediator():
    optimizer=AlgorithmicRecourse(chain(),immutable_features=['M'])
    plans=optimizer.compute_recourse({'X':0,'M':0,'Y':0},'Y',1,.8)
    assert plans and plans[0].interventions=={'X':1}
    assert plans[0].target_probability==pytest.approx(.86)
    assert plans[0].baseline_probability==pytest.approx(.14)

def test_immutable_and_no_feasible_plan():
    plans=AlgorithmicRecourse(chain(),immutable_features=['X','M']).compute_recourse({'X':0,'M':0,'Y':0},'Y',1,.9)
    assert plans==[]

def test_no_action_when_goal_already_met():
    plans=AlgorithmicRecourse(chain()).compute_recourse({'X':1,'M':1,'Y':1},'Y',1,.8)
    assert plans[0].interventions=={} and plans[0].total_cost==0

def test_adjustment_blocks_confounder():
    n=chain();n.add_node('Z');n.add_edge('Z','X');n.add_edge('Z','Y')
    assert InterventionAnalysis(n).find_adjustment_set('X','Y')=={'Z'}

def test_collider_does_not_need_adjustment():
    n=CausalBayesianNetwork();n.add_nodes(['X','Y','C']);n.add_edges([('X','C'),('Y','C')])
    assert InterventionAnalysis(n).find_adjustment_set('X','Y')==set()

def test_d_separation_current_networkx():
    assert chain().is_d_separated(['X'],['Y'],['M'])

def test_shd_reversal_is_one_edit():
    d=CausalDiscovery(pd.DataFrame({'X':[0,1]*20,'Y':[0,1]*20}))
    d.learned_dag=CausalBayesianNetwork();d.learned_dag.add_nodes(['X','Y']);d.learned_dag.add_edge('Y','X')
    truth=nx.DiGraph([('X','Y')]);assert d.evaluate_structure(truth)['structural_hamming_distance']==1

def test_preprocessing_uses_only_training_boundaries():
    train=pd.DataFrame({'x':range(100),'target':[0,1]*50})
    p=Preprocessor().fit(train,'target');cuts=p.specs['x']['cuts'].copy()
    transformed=p.transform(pd.DataFrame({'x':[999999],'target':[1]}))
    assert p.specs['x']['cuts']==cuts and transformed['x'].iloc[0]==2

def test_unseen_category_is_actionable_error():
    p=Preprocessor().fit(pd.DataFrame({'x':['a','b'],'y':[0,1]}),'y')
    with pytest.raises(ValueError,match='only in test'):p.transform(pd.DataFrame({'x':['c'],'y':[0]}))

@pytest.fixture(scope='module')
def service():return StudioService()

@pytest.mark.parametrize('domain',['fintech','healthcare','ecommerce','education','hr'])
def test_all_domains_end_to_end(service,domain):
    m=service.train(domain=domain,samples=500)
    assert m['train_rows']==400 and m['test_rows']==100
    assert len(m['evaluation'])==3
    assert sum(map(sum,m['confusion']))==100
    assert service.get(m['id'])['network'].validate()
    n=m['treatment'];result=service.intervene(m['id'],n,m['states'][n][-1],m['states'][n][0],m['target'],m['desired'])
    assert sum(result['interventional'].values())==pytest.approx(1)
    assert -1<=result['effect']<=1
    plans=service.recourse(m['id'],m['sample_profiles'][0],m['target'],m['desired'],m['immutable'],.65)['plans']
    for p in plans: assert not set(p['interventions']).intersection(m['immutable'])

@pytest.mark.parametrize('algorithm',['pc','tan','reference'])
def test_learning_modes(service,algorithm):
    m=service.train(domain='healthcare',samples=500,algorithm=algorithm)
    assert m['metrics']['is_dag']
    if algorithm=='reference':assert m['metrics']['structural_hamming_distance']==0

def test_custom_text_csv(service):
    records=[{'Group':['A','B'][i%2],'Effort':['low','high'][i%2],'Result':['fail','pass'][i%2]} for i in range(200)]
    m=service.train(domain='custom',records=records,target='Result')
    assert m['reference_edges'] is None and set(m['labels']['Result'].values())=={'fail','pass'}
    assert m['evaluation'][0]['accuracy']>.9

def test_same_size_uploads_have_different_sessions(service):
    a=[{'X':i%2,'Y':i%2} for i in range(200)]
    b=[{'X':i%2,'Y':(i//2)%2} for i in range(200)]
    assert service.train(records=a,target='Y')['id']!=service.train(records=b,target='Y')['id']

def test_api_validation_and_static_frontend():
    client=TestClient(app)
    assert client.get('/api/health').json()['frontend']=='React'
    assert client.get('/').status_code==200
    assert client.post('/api/train',json={'samples':1}).status_code==422
    assert client.post('/api/train',json={'domain':'unknown'}).status_code==422
    assert client.post('/api/intervene',json={'session_id':'missing','treatment':'X','value':1,'control':0,'target':'Y','desired':1}).status_code==422
