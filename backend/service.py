"""Train-only preprocessing, reproducible evaluation, and bounded model sessions."""
from __future__ import annotations
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import asdict
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix
from pgmpy.inference import VariableElimination
from src.data.benchmarks import (generate_credit_risk_dataset, generate_healthcare_dataset,
    generate_ecommerce_churn_dataset, generate_education_dataset, generate_hr_attrition_dataset)
from src.discovery.structure_learning import CausalDiscovery
from src.models.parameter_learning import ParameterLearning
from src.recourse.recourse_optimizer import AlgorithmicRecourse

DOMAINS = {
 'fintech': (generate_credit_risk_dataset, 'Approval', 1, 'Credit intelligence', ['Age'], 'Savings'),
 'healthcare': (generate_healthcare_dataset, 'Recovery', 1, 'Clinical outcomes', ['Age'], 'Treatment'),
 'ecommerce': (generate_ecommerce_churn_dataset, 'Churn', 0, 'Customer retention', ['Tenure'], 'Discount_Offered'),
 'education': (generate_education_dataset, 'Exam_Pass', 1, 'Student success', ['Parent_Education'], 'Study_Hours'),
 'hr': (generate_hr_attrition_dataset, 'Promoted', 1, 'Workforce growth', ['Experience_Level'], 'Training_Completed'),
}
LABELS = {
 'Age': ['Young', 'Middle', 'Senior'], 'Income': ['Low','Medium','High'],
 'Savings': ['Low','Medium','High'], 'Loan_Amount': ['Small','Medium','Large'],
 'Credit_Score': ['Poor','Fair','Excellent'], 'Installment_Rate':['Low','High'],
 'Approval':['Rejected','Approved'], 'Disease_Severity':['Mild','Severe'],
 'Treatment':['Standard','Aggressive'], 'Biomarker_A':['Normal','Elevated'],
 'Recovery':['Deteriorated','Recovered'], 'Tenure':['New','Long-term'],
 'Support_Calls':['Low','High'], 'Monthly_Spend':['Low','High'],
 'Discount_Offered':['No','Yes'], 'Churn':['Stayed','Churned'],
 'Parent_Education':['High school','College','Postgraduate'], 'Attendance':['Low','High'],
 'Study_Hours':['Low','Medium','High'], 'Tutoring':['No','Yes'], 'Exam_Pass':['Failed','Passed'],
 'Experience_Level':['Junior','Mid','Senior'], 'Overtime':['Normal','High'],
 'Training_Completed':['None','Advanced'], 'Performance_Rating':['Low','High'], 'Promoted':['No','Yes']}

def clean_json(value):
    return json.loads(json.dumps(value, default=lambda v: v.item() if hasattr(v, 'item') else list(v), allow_nan=False))

class Preprocessor:
    """Fit category vocabularies and quantile boundaries using training rows only."""
    def fit(self, frame, target):
        self.specs = {}
        for col in frame:
            series = frame[col]
            if series.isna().any():
                raise ValueError(f'{col}: missing values must be filled or removed before upload.')
            if col != target and pd.api.types.is_numeric_dtype(series) and series.nunique() > 6:
                cuts = np.unique(np.quantile(series.astype(float), [1/3, 2/3])).tolist()
                self.specs[col] = {'kind':'bins','cuts':cuts,'labels':[f'Band {i+1}' for i in range(len(cuts)+1)]}
            else:
                values = sorted(series.unique().tolist(), key=lambda x: str(x))
                if len(values) > 8:
                    raise ValueError(f'{col}: use at most 8 categories; remove IDs or choose a categorical target.')
                self.specs[col] = {'kind':'category','values':values,'labels':[str(x) for x in values]}
        return self

    def transform(self, frame):
        result = pd.DataFrame(index=frame.index)
        for col, spec in self.specs.items():
            if spec['kind'] == 'bins':
                result[col] = np.searchsorted(spec['cuts'], frame[col].astype(float), side='right')
            else:
                mapping = {v:i for i,v in enumerate(spec['values'])}
                result[col] = frame[col].map(mapping)
                if result[col].isna().any():
                    raise ValueError(f'{col}: a category occurs only in test rows. Add examples or combine rare categories.')
                result[col] = result[col].astype(int)
        return result

class StudioService:
    def __init__(self):
        self.sessions = OrderedDict()

    def get(self, sid):
        if sid not in self.sessions:
            raise ValueError('This workspace expired. Train the model again.')
        self.sessions.move_to_end(sid)
        return self.sessions[sid]

    def train(self, domain='fintech', algorithm='hill', samples=2000, seed=42, records=None, target=None):
        started = time.perf_counter()
        if algorithm not in ('hill','pc','tan','reference'):
            raise ValueError('Choose Hill-Climbing, PC, TAN, or reference graph.')
        truth = None
        if records is None:
            if domain not in DOMAINS:
                raise ValueError('Unknown benchmark domain.')
            generator, target, desired_raw, title, immutable, treatment = DOMAINS[domain]
            raw, truth = generator(n_samples=samples, seed=seed)
        else:
            raw = pd.DataFrame(records)
            title, immutable, treatment, desired_raw = 'Custom dataset', [], None, None
            if not 100 <= len(raw) <= 10000 or not 2 <= len(raw.columns) <= 12:
                raise ValueError('Upload 100–10,000 rows with 2–12 columns.')
            if target not in raw or raw[target].nunique() < 2:
                raise ValueError('Choose a target column with at least two states.')
            if raw.isna().any().any() or raw.replace([np.inf,-np.inf], np.nan).isna().any().any():
                raise ValueError('The dataset contains missing or infinite values. Clean it before upload.')
        if algorithm == 'reference' and truth is None:
            raise ValueError('Reference graphs are available only for synthetic benchmarks.')
        if raw[target].value_counts().min() < 5:
            raise ValueError('Each outcome category needs at least five rows for a stratified split.')
        train_raw, test_raw = train_test_split(raw, test_size=0.2, random_state=seed, stratify=raw[target])
        prep = Preprocessor().fit(train_raw, target)
        train, test = prep.transform(train_raw), prep.transform(test_raw)
        data = prep.transform(raw)
        discovery = CausalDiscovery(train)
        if algorithm == 'hill':
            network = discovery.discover_hill_climb(max_iter=100)
        elif algorithm == 'pc':
            network = discovery.discover_pc(max_cond_vars=3)
        elif algorithm == 'tan':
            network = discovery.discover_tree(class_node=target)
        else:
            from src.models.bayesian_network import CausalBayesianNetwork
            network = CausalBayesianNetwork()
            network.add_nodes(list(train.columns))
            network.add_edges(list(truth.edges()))
            discovery.learned_dag = network
        state_counts = {c: train[c].nunique() for c in train}
        cells = sum(state_counts[n] * int(np.prod([state_counts[p] for p in network.get_parents(n)])) for n in train)
        moral = nx.moral_graph(network.dag)
        width, _ = nx.approximation.treewidth_min_fill_in(moral)
        if cells > 100000 or max(state_counts.values()) ** (width + 1) > 2000000:
            raise ValueError('This graph is too complex for exact inference. Reduce columns or categories, or use TAN.')
        ParameterLearning(network).fit_bayesian(train)
        metrics = discovery.evaluate_structure(truth)
        inference = VariableElimination(network.model)
        features = [c for c in train if c != target]
        states = list(network.cpds[target].state_names[target])
        memo = {}
        probabilities = []
        for row in test[features].to_dict('records'):
            key = tuple(row.items())
            if key not in memo:
                q = inference.query([target], evidence=row, show_progress=False)
                memo[key] = [float(q.values[q.state_names[target].index(s)]) for s in states]
            probabilities.append(memo[key])
        probabilities = np.array(probabilities)
        y = test[target].to_numpy()
        prediction = np.array(states)[probabilities.argmax(axis=1)]
        rf = RandomForestClassifier(n_estimators=100, max_depth=7, min_samples_leaf=5, random_state=seed, n_jobs=1)
        rf.fit(train[features], train[target])
        rf_probs = rf.predict_proba(test[features])
        majority = int(train[target].mode().iloc[0])
        majority_pred = np.full(len(test), majority)
        prior = train[target].value_counts(normalize=True)
        prior_probs = np.tile([prior.get(s, 0) for s in states], (len(test),1))
        def scores(name, pred, probs):
            return {'model':name, 'accuracy':accuracy_score(y,pred),
                    'f1':f1_score(y,pred,average='macro',zero_division=0),
                    'log_loss':log_loss(y,probs,labels=states)}
        evaluation = [scores('Bayesian network',prediction,probabilities),
                      scores('Random forest',rf.predict(test[features]),rf_probs),
                      scores('Majority baseline',majority_pred,prior_probs)]
        labels = {}
        for col, spec in prep.specs.items():
            label_list = spec['labels']
            if records is None and col in LABELS:
                label_list = [LABELS[col][int(v)] for v in spec.get('values', [])]
                if domain == 'healthcare' and col == 'Age': label_list=['Young','Elderly']
            labels[col] = {str(i):label for i,label in enumerate(label_list)}
        desired = (prep.specs[target]['values'].index(desired_raw) if desired_raw is not None else states[-1])
        sid = hashlib.sha256(pd.util.hash_pandas_object(raw,index=True).values.tobytes()
                             + json.dumps([list(raw.columns),target,algorithm,seed]).encode()).hexdigest()[:20]
        session = {'network':network,'data':data,'raw':raw,'prep':prep,'labels':labels,
                   'target':target, 'desired':desired,'truth':truth,'evaluation':evaluation}
        self.sessions[sid] = session
        self.sessions.move_to_end(sid)
        while len(self.sessions)>8: self.sessions.popitem(last=False)
        result = {'id':sid,'domain':domain,'title':title,'algorithm':algorithm,'target':target,'desired':desired,
                  'immutable':immutable,'treatment':treatment or features[0], 'labels':labels,
                  'states':{c:list(network.cpds[c].state_names[c]) for c in data},
                  'rows':len(raw),'train_rows':len(train),'test_rows':len(test),'seed':seed,
                  'metrics':metrics,'evaluation':evaluation,'confusion':confusion_matrix(y,prediction,labels=states).tolist(),
                  'nodes':list(data.columns),'edges':list(network.dag.edges()),
                  'reference_edges':list(truth.edges()) if truth is not None else None,
                  'preview':data.head(100).to_dict('records'),'sample_profiles':data.head(30).to_dict('records'),
                  'distribution':{c:{str(k):int(v) for k,v in data[c].value_counts().items()} for c in data},
                  'preprocessing':prep.specs, 'seconds':round(time.perf_counter()-started,2),
                  'source':'Synthetic benchmark' if records is None else 'User-provided CSV'}
        session['summary'] = clean_json(result)
        return session['summary']

    def intervene(self, sid, treatment, value, control, target, desired):
        from src.inference.intervention import InterventionAnalysis
        s = self.get(sid); engine = InterventionAnalysis(s['network'])
        if treatment == target: raise ValueError('Treatment and outcome must be different.')
        if desired not in s['network'].cpds[target].state_names[target]: raise ValueError('Invalid desired state.')
        observed = engine.do_intervention({},target,{treatment:value})
        action = engine.do_intervention({treatment:value},target)
        baseline = engine.do_intervention({treatment:control},target)
        adjustment = engine.find_adjustment_set(treatment,target)
        return clean_json({'observational':observed,'interventional':action,'control':baseline,
                           'effect':action[str(desired)]-baseline[str(desired)],
                           'adjustment':sorted(adjustment) if adjustment is not None else None,
                           'removed_edges':list(s['network'].dag.in_edges(treatment))})

    def recourse(self, sid, profile, target, desired, immutable, threshold, max_actions=2, costs=None):
        s=self.get(sid)
        if target not in s['network'].cpds or desired not in s['network'].cpds[target].state_names[target]:
            raise ValueError('Choose a valid outcome and desired state.')
        if set(profile) != set(s['data'].columns): raise ValueError('Supply every profile variable.')
        for node,value in profile.items():
            if value not in s['network'].cpds[node].state_names[node]: raise ValueError(f'Invalid profile state for {node}.')
        if not set(immutable).issubset(profile): raise ValueError('Unknown immutable feature.')
        optimizer=AlgorithmicRecourse(s['network'],immutable_features=immutable,action_costs=costs)
        plans=optimizer.compute_recourse(profile,target,desired,threshold,max_interventions=max_actions,top_k=3)
        return clean_json({'plans':[asdict(p) for p in plans],
            'meaning':'Conditional intervention scenarios, not identified individual counterfactuals. Descendants are allowed to respond to actions.'})
