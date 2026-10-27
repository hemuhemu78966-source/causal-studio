"""
Generate 5-Section PowerPoint Presentation (.pptx) matching exact academic structure.
Sections:
1. TITLE
2. ABSTRACT
3. INTRODUCTION
4. LITERATURE SURVEY
5. MERITS AND DEMERITS / CONCLUSION
"""

import os
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

def build_presentation(output_paths):
    prs = Presentation()
    prs.slide_width = Inches(13.333)  # 16:9 widescreen
    prs.slide_height = Inches(7.5)

    PRIMARY = RGBColor(30, 58, 138)     # Navy Blue
    SECONDARY = RGBColor(37, 99, 235)   # Royal Blue
    DARK = RGBColor(30, 41, 59)         # Dark Slate
    MUTED = RGBColor(100, 116, 139)     # Gray
    LIGHT_BG = RGBColor(248, 250, 252)   # Card bg
    BORDER = RGBColor(226, 232, 240)    # Border
    GREEN = RGBColor(22, 101, 52)       # Green
    RED = RGBColor(185, 28, 28)         # Red
    WHITE = RGBColor(255, 255, 255)

    blank_layout = prs.slide_layouts[6]

    def add_section_header(slide, section_tag, main_title):
        box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(1.1))
        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        p_sec = tf.paragraphs[0]
        p_sec.text = section_tag.upper()
        p_sec.font.size = Pt(11)
        p_sec.font.bold = True
        p_sec.font.color.rgb = SECONDARY

        p_title = tf.add_paragraph()
        p_title.text = main_title
        p_title.font.size = Pt(23)
        p_title.font.bold = True
        p_title.font.color.rgb = PRIMARY

    # =========================================================================
    # 1. TITLE
    # =========================================================================
    slide1 = prs.slides.add_slide(blank_layout)
    bg1 = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = PRIMARY
    bg1.line.fill.background()

    tb1 = slide1.shapes.add_textbox(Inches(1.0), Inches(1.2), Inches(11.3), Inches(5.2))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p = tf1.paragraphs[0]
    p.text = "1. TITLE OF THE PROJECT"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = RGBColor(191, 219, 254)

    p_t = tf1.add_paragraph()
    p_t.text = "CAUSAL STUDIO: EXPLAINABLE CAUSAL DECISION INTELLIGENCE & ALGORITHMIC RECOURSE PLATFORM"
    p_t.font.size = Pt(32)
    p_t.font.bold = True
    p_t.font.color.rgb = WHITE
    p_t.space_before = Pt(8)

    p_sub = tf1.add_paragraph()
    p_sub.text = "Using Causal Bayesian Networks and Structural Causal Models for Actionable Decision Support"
    p_sub.font.size = Pt(18)
    p_sub.font.color.rgb = RGBColor(226, 232, 240)
    p_sub.space_before = Pt(10)

    p_meta = tf1.add_paragraph()
    p_meta.text = (
        "\n• Student Name: [Your Name]          • Registration Number: [Your Roll Number]\n"
        "• Faculty Guide: [Faculty Name]      • Course Title: Artificial Intelligence / Capstone\n"
        "• Department of Computer Science & Engineering | Date of Submission: September 2026"
    )
    p_meta.font.size = Pt(13)
    p_meta.font.color.rgb = RGBColor(203, 213, 225)
    p_meta.space_before = Pt(30)

    # =========================================================================
    # 2. ABSTRACT
    # =========================================================================
    slide2 = prs.slides.add_slide(blank_layout)
    add_section_header(slide2, "2. Abstract", "Summary of the AI Technology, Importance & Findings")

    # Abstract Content Box
    ab_box = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.1))
    ab_box.fill.solid()
    ab_box.fill.fore_color.rgb = LIGHT_BG
    ab_box.line.color.rgb = BORDER
    tf_ab = ab_box.text_frame
    tf_ab.word_wrap = True

    p = tf_ab.paragraphs[0]
    p.text = "Executive Summary:"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = PRIMARY

    p_txt = tf_ab.add_paragraph()
    p_txt.text = (
        "Conventional Machine Learning models operate strictly on statistical associations P(Y|X), rendering them vulnerable to "
        "spurious correlations, confounding bias (Simpson's Paradox), and an inability to simulate active interventions or counterfactual 'what-if' queries. "
        "This project introduces Causal Studio, an end-to-end Causal Decision Intelligence and Algorithmic Recourse platform grounded in Judea Pearl's "
        "Structural Causal Models (SCMs) and do-calculus P(Y|do(X)).\n\n"
        "The platform automatically discovers causal Directed Acyclic Graphs (DAGs) from raw tabular CSV data using constraint-based (PC algorithm) "
        "and score-based (Hill-Climbing with BIC) methods, fits probability distributions using Bayesian Dirichlet priors, and resolves confounding via automated "
        "backdoor adjustment. Additionally, it implements a prescriptive Algorithmic Recourse Optimizer that computes minimal-cost actionable feature interventions "
        "to reverse unfavorable outcomes while strictly respecting immutable features (e.g., Age) and directional domain constraints."
    )
    p_txt.font.size = Pt(13)
    p_txt.font.color.rgb = DARK
    p_txt.space_before = Pt(6)

    p_kw = tf_ab.add_paragraph()
    p_kw.text = "Keywords: Causal Bayesian Networks (CBN), Structural Causal Models (SCM), Do-Calculus, Algorithmic Recourse, Causal Discovery, PC Algorithm, Counterfactual Inference, Explainable AI (XAI)."
    p_kw.font.bold = True
    p_kw.font.size = Pt(12.5)
    p_kw.font.color.rgb = SECONDARY
    p_kw.space_before = Pt(14)

    # =========================================================================
    # 3. INTRODUCTION - 3.1 Background & Importance
    # =========================================================================
    slide3 = prs.slides.add_slide(blank_layout)
    add_section_header(slide3, "3. Introduction", "3.1 Background of Technology & Why This Topic is Important")

    box_b1 = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.1))
    box_b1.fill.solid()
    box_b1.fill.fore_color.rgb = LIGHT_BG
    box_b1.line.color.rgb = BORDER
    tf_b1 = box_b1.text_frame
    tf_b1.word_wrap = True
    p = tf_b1.paragraphs[0]
    p.text = "Background of Causal AI"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = PRIMARY

    b_pts = [
        "Judea Pearl's Ladder of Causation defines 3 operational tiers:",
        "Layer 1 (Association): P(Y|X) - Passive correlation & standard ML curve-fitting.",
        "Layer 2 (Intervention): P(Y|do(X)) - Simulating active policy actions via graph mutilation.",
        "Layer 3 (Counterfactuals): P(Y_x'|X=x, Y=y) - Retrospective what-if reasoning.",
        "Standard ML stops at Layer 1; Causal Studio navigates all 3 tiers."
    ]
    for pt in b_pts:
        p_pt = tf_b1.add_paragraph()
        p_pt.text = f"• {pt}"
        p_pt.font.size = Pt(12.5)
        p_pt.font.color.rgb = DARK
        p_pt.space_before = Pt(8)

    box_b2 = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.1))
    box_b2.fill.solid()
    box_b2.fill.fore_color.rgb = RGBColor(239, 246, 255)
    box_b2.line.color.rgb = RGBColor(191, 219, 254)
    tf_b2 = box_b2.text_frame
    tf_b2.word_wrap = True
    p = tf_b2.paragraphs[0]
    p.text = "Why This Topic is Critical?"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = SECONDARY

    imp_pts = [
        "Overcoming Simpson's Paradox: Confounding variables can reverse statistical signs, causing ML models to make harmful recommendations in clinical/financial decisions.",
        "Actionable Recourse vs Static Explainability: Standard XAI (SHAP/LIME) only explains why an event occurred; Algorithmic Recourse prescribes exact actions to achieve success.",
        "Safety & Ethical Compliance: Enforces protection of immutable characteristics (Age, Gender) and directional feasibility in high-stakes automated decisions."
    ]
    for pt in imp_pts:
        p_pt = tf_b2.add_paragraph()
        p_pt.text = f"• {pt}"
        p_pt.font.size = Pt(12.5)
        p_pt.font.color.rgb = DARK
        p_pt.space_before = Pt(10)

    # =========================================================================
    # 3. INTRODUCTION - 3.2 System Architecture
    # =========================================================================
    slide4 = prs.slides.add_slide(blank_layout)
    add_section_header(slide4, "3. Introduction", "3.2 Architecture Diagram & End-to-End Decision Flow")

    arch_steps = [
        ("Layer 1: Ingestion & Discovery", "• Tabular CSV Data Ingestion\n• PC Constraint Discovery\n• Hill-Climbing BIC Search\n• Structural Hamming Dist (SHD)", Inches(0.8)),
        ("Layer 2: Parameterization", "• Bayesian Dirichlet Priors\n• Maximum Likelihood (MLE)\n• Non-Linear SCM ANMs\n• Topological Sort", Inches(3.8)),
        ("Layer 3: Causal Inference", "• Do-Calculus Mutilation\n• Backdoor Confounder Sets\n• ATE / CATE Estimation\n• Recourse Cost Optimizer", Inches(6.8)),
        ("Layer 4: Interactive Studio", "• Physics-Based DAG (PyVis)\n• Real-Time Intervention Sliders\n• Recourse Action Cards\n• 5 Industry Benchmarks", Inches(9.8))
    ]

    for stitle, scontent, x_pos in arch_steps:
        sbox = slide4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x_pos, Inches(1.6), Inches(2.7), Inches(5.1))
        sbox.fill.solid()
        sbox.fill.fore_color.rgb = LIGHT_BG
        sbox.line.color.rgb = SECONDARY
        sbox.line.width = Pt(1.5)
        tf_s = sbox.text_frame
        tf_s.word_wrap = True
        p = tf_s.paragraphs[0]
        p.text = stitle
        p.font.bold = True
        p.font.size = Pt(14)
        p.font.color.rgb = PRIMARY

        p2 = tf_s.add_paragraph()
        p2.text = scontent
        p2.font.size = Pt(12.5)
        p2.font.color.rgb = DARK
        p2.space_before = Pt(14)

    # =========================================================================
    # 3. INTRODUCTION - 3.3 Methodologies Used
    # =========================================================================
    slide5 = prs.slides.add_slide(blank_layout)
    add_section_header(slide5, "3. Introduction", "3.3 Methodologies Used Across the Platform")

    meth_cards = [
        ("1. Causal Discovery (PC & Hill-Climb)", "• PC Algorithm: Uses Chi-Square tests to prune edges and Meek's rules to orient colliders.\n• Hill-Climb: Greedily optimizes BIC score: log P(D|theta) - (d/2)*log(N).", Inches(0.8), Inches(1.6)),
        ("2. Do-Calculus & Backdoor Adjustment", "• Graph Mutilation: Severs parent edges into intervention node X and sets P(X=x)=1.0.\n• Backdoor Adjustment: Identifies minimal adjustment set Z to isolate causal ATE/CATE.", Inches(6.8), Inches(1.6)),
        ("3. Prescriptive Algorithmic Recourse", "• Formulates constrained optimization: min sum w_i * Cost(x_i, x_i*) subject to P(Y=1|do(X*)) >= tau.\n• Enforces immutable feature locks (Age) and directional monotonicity.", Inches(0.8), Inches(4.2)),
        ("4. Continuous Non-Linear SCMs", "• Additive Noise Models: X_j = f_j(Parents) + U_j.\n• 3-Step Pearl Counterfactuals: Noise Abduction -> Equation Action -> Forward Prediction.", Inches(6.8), Inches(4.2))
    ]

    for mtitle, mbody, mx, my in meth_cards:
        mbox = slide5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, mx, my, Inches(5.7), Inches(2.4))
        mbox.fill.solid()
        mbox.fill.fore_color.rgb = LIGHT_BG
        mbox.line.color.rgb = BORDER
        tf_m = mbox.text_frame
        tf_m.word_wrap = True
        p = tf_m.paragraphs[0]
        p.text = mtitle
        p.font.bold = True
        p.font.size = Pt(14.5)
        p.font.color.rgb = PRIMARY

        p2 = tf_m.add_paragraph()
        p2.text = mbody
        p2.font.size = Pt(12)
        p2.font.color.rgb = DARK
        p2.space_before = Pt(6)

    # =========================================================================
    # 4. LITERATURE SURVEY - Table 1
    # =========================================================================
    slide6 = prs.slides.add_slide(blank_layout)
    add_section_header(slide6, "4. Literature Survey", "Recent Research Papers (2023 - 2026) - Part 1")

    # Table
    t_shape1 = slide6.shapes.add_table(6, 5, Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.1))
    t1 = t_shape1.table
    t1.columns[0].width = Inches(2.2)
    t1.columns[1].width = Inches(2.5)
    t1.columns[2].width = Inches(2.2)
    t1.columns[3].width = Inches(2.4)
    t1.columns[4].width = Inches(2.4)

    headers = ["Author & Year", "Paper Title", "Methodologies Used", "Merits", "Demerits"]
    for i, h in enumerate(headers):
        cell = t1.cell(0, i)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = PRIMARY
        for p in cell.text_frame.paragraphs:
            p.font.bold = True
            p.font.size = Pt(11)
            p.font.color.rgb = WHITE

    papers1 = [
        ("Karimi et al. (2024)\nACM CSUR", "A Survey on Algorithmic Recourse", "SCM action search, graph traversal, cost optimization", "Bridges correlation XAI to causal actions; formalizes feasibility.", "Assumes full knowledge of true SCM; high computation on large DAGs."),
        ("Pearl & Bareinboim (2024)\nNature Machine Intelligence", "External Validity & Generalizability", "Do-calculus, transportability theory, selection diagrams", "Rigorous mathematical bounds for cross-domain knowledge transfer.", "Requires explicit modeling of cross-domain discrepancy nodes."),
        ("Sanchez et al. (2024)\nLancet Digital Health", "Causal ML for Healthcare", "Backdoor adjustment, ATE/CATE estimation in EHR records", "Eliminates treatment assignment bias in multi-center clinical trials.", "Susceptible to positivity violations on missing medical records."),
        ("Mahajan et al. (2024)\nICML 2024", "Recourse Under Causal Uncertainty", "Bounding counterfactual probabilities, robust recourse", "Provides recourse bounds when the causal DAG is only partially known.", "Yields conservative recommendations with higher action costs."),
        ("Vowels et al. (2023)\nACM Computing Surveys", "D'ya Like DAGs? Practical Guide", "Comparative benchmark of NOTEARS, PC, and Hill-Climb", "Comprehensive speed and graph recovery benchmarks across data types.", "Continuous search (NOTEARS) can yield cyclic errors in finite samples.")
    ]

    for r_idx, r_data in enumerate(papers1, 1):
        for c_idx, txt in enumerate(r_data):
            cell = t1.cell(r_idx, c_idx)
            cell.text = txt
            cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT_BG if r_idx % 2 == 0 else WHITE
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(10)
                p.font.color.rgb = DARK

    # =========================================================================
    # 4. LITERATURE SURVEY - Table 2
    # =========================================================================
    slide7 = prs.slides.add_slide(blank_layout)
    add_section_header(slide7, "4. Literature Survey", "Recent Research Papers (2023 - 2026) - Part 2")

    t_shape2 = slide7.shapes.add_table(6, 5, Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.1))
    t2 = t_shape2.table
    t2.columns[0].width = Inches(2.2)
    t2.columns[1].width = Inches(2.5)
    t2.columns[2].width = Inches(2.2)
    t2.columns[3].width = Inches(2.4)
    t2.columns[4].width = Inches(2.4)

    for i, h in enumerate(headers):
        cell = t2.cell(0, i)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = PRIMARY
        for p in cell.text_frame.paragraphs:
            p.font.bold = True
            p.font.size = Pt(11)
            p.font.color.rgb = WHITE

    papers2 = [
        ("Pawlowski et al. (2024)\nIEEE TPAMI", "Deep Structural Causal Models", "Normalizing flows, variational SCMs, counterfactual abduction", "Synthesizes high-dimensional counterfactual images/tabular records.", "Complex training dynamics; requires vast data to avoid posterior collapse."),
        ("Chernozhukov et al. (2024)\nEconometrica", "Double/Debiased Machine Learning", "Neyman orthogonality, cross-fitting, high-dim confounder control", "Removes regularization bias, enabling valid causal confidence intervals.", "Requires large sample sizes for cross-fitting convergence."),
        ("Curth & van der Schaar (2023)\nICLR 2023", "Counterfactual Reasoning in Deep Estimators", "Inductive biases in treatment architectures (TARNet)", "Theoretical bounds on when representation sharing improves treatment effects.", "Primarily focuses on binary treatment setups; hard to scale to continuous."),
        ("Sharma & Kiciman (2023)\nJMLR 2023", "DoWhy: Causal Inference & Refutation", "4-step causal protocol: Model, Identify, Estimate, Refute", "Establishes refutation tests (placebo treatments) to validate claims.", "Refutations indicate violations but do not auto-correct the graph structure."),
        ("Chattopadhyay et al. (2025)\nIEEE TNNLS", "Neural Causal Recourse for Credit", "Tabular recourse optimization, credit scoring compliance", "Directly satisfies fair-lending regulatory non-discrimination rules.", "Search complexity scales exponentially with large discrete bins.")
    ]

    for r_idx, r_data in enumerate(papers2, 1):
        for c_idx, txt in enumerate(r_data):
            cell = t2.cell(r_idx, c_idx)
            cell.text = txt
            cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT_BG if r_idx % 2 == 0 else WHITE
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(10)
                p.font.color.rgb = DARK

    # =========================================================================
    # 5. MERITS AND DEMERITS
    # =========================================================================
    slide8 = prs.slides.add_slide(blank_layout)
    add_section_header(slide8, "5. Merits and Demerits", "Critical Evaluation of the Causal Studio Platform")

    box_mer = slide8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.1))
    box_mer.fill.solid()
    box_mer.fill.fore_color.rgb = RGBColor(240, 253, 244)
    box_mer.line.color.rgb = RGBColor(187, 247, 208)
    tf_mer = box_mer.text_frame
    tf_mer.word_wrap = True
    p = tf_mer.paragraphs[0]
    p.text = "Platform Merits (Strengths)"
    p.font.bold = True
    p.font.size = Pt(17)
    p.font.color.rgb = GREEN

    mer_list = [
        "Confounder Elimination: Eliminates Simpson's Paradox and spurious correlations via formal do-calculus.",
        "Prescriptive Actionability: Generates concrete, cost-optimal intervention paths rather than static importance scores.",
        "Safety & Ethical Feasibility: Strictly protects immutable variables (Age, Gender) and enforces directional constraints (Experience).",
        "End-to-End Workflow: Unified pipeline from raw CSV discovery to Bayesian parameter estimation and interactive visual recourse."
    ]
    for m in mer_list:
        p_m = tf_mer.add_paragraph()
        p_m.text = f"✓ {m}"
        p_m.font.size = Pt(12.5)
        p_m.font.color.rgb = DARK
        p_m.space_before = Pt(10)

    box_dem = slide8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.1))
    box_dem.fill.solid()
    box_dem.fill.fore_color.rgb = RGBColor(254, 242, 242)
    box_dem.line.color.rgb = RGBColor(254, 202, 202)
    tf_dem = box_dem.text_frame
    tf_dem.word_wrap = True
    p = tf_dem.paragraphs[0]
    p.text = "Platform Demerits (Limitations & Scope)"
    p.font.bold = True
    p.font.size = Pt(17)
    p.font.color.rgb = RED

    dem_list = [
        "Causal Sufficiency Assumption: Assumes all relevant confounding variables are observed in the tabular dataset.",
        "Combinatorial Search Complexity: Exact DAG search is NP-hard; heuristic search scales super-linearly when variables > 50.",
        "Discretization Trade-off: Tabular discrete Bayesian networks require binning continuous data, causing minor information loss.",
        "Sample Size Dependency: Conditional independence tests (Chi-Square) require sufficient sample size across conditioning subsets."
    ]
    for d in dem_list:
        p_d = tf_dem.add_paragraph()
        p_d.text = f"⚠ {d}"
        p_d.font.size = Pt(12.5)
        p_d.font.color.rgb = DARK
        p_d.space_before = Pt(10)

    # Save to paths
    for path in output_paths:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            prs.save(path)
            print(f"Updated 5-Section Presentation saved to: {path}")
        except Exception as e:
            print(f"Could not overwrite {path} (it may be open in PowerPoint): {e}")

if __name__ == "__main__":
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "Causal_Studio_Presentation.pptx")
    project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Causal_Studio_Presentation.pptx")
    build_presentation([desktop_path, project_path])
