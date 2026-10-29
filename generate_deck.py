import os
import numpy as np
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# 16:9 Widescreen slide format: 960 x 540 points
SLIDE_WIDTH = 960
SLIDE_HEIGHT = 540
PAGE_SIZE = (SLIDE_WIDTH, SLIDE_HEIGHT)

class SlideCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pages = []

    def showPage(self):
        self.pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self.pages)
        for page in self.pages:
            self.__dict__.update(page)
            self.draw_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_decorations(self, total_pages):
        # Draw background and header/footer on non-title slides
        if self._pageNumber > 1:
            # Subtle header top accent bar
            self.setFillColor(colors.HexColor("#0284c7"))
            self.rect(0, SLIDE_HEIGHT - 4, SLIDE_WIDTH, 4, fill=1, stroke=0)

            # Footer bar
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.8)
            self.line(40, 32, SLIDE_WIDTH - 40, 32)

            # Footer texts
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor("#64748b"))
            self.drawString(40, 20, "TCP-RL-ns3 | Autonomous TCP Congestion Control via PPO in ns-3")

            self.setFont("Helvetica", 8)
            self.drawString(SLIDE_WIDTH / 2 - 40, 20, "github.com/shetgaonkarshubh/tcp-rl-ns3")

            page_str = f"{self._pageNumber} / {total_pages}"
            self.drawRightString(SLIDE_WIDTH - 40, 20, page_str)


def create_slide_header(category: str, title: str, subtitle: str, styles):
    content = []
    
    # Category badge
    cat_style = ParagraphStyle(
        "CatStyle",
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=11,
        textColor=colors.HexColor("#0284c7")
    )
    content.append(Paragraph(category.upper(), cat_style))
    content.append(Spacer(1, 2))

    # Main Slide Title
    title_style = ParagraphStyle(
        "SlideTitleStyle",
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=23,
        textColor=colors.HexColor("#0f172a")
    )
    content.append(Paragraph(title, title_style))
    content.append(Spacer(1, 2))

    # Subtitle
    sub_style = ParagraphStyle(
        "SlideSubStyle",
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#64748b")
    )
    content.append(Paragraph(subtitle, sub_style))
    content.append(Spacer(1, 14))

    return content


def make_card(header_text: str, body_paragraphs: list, styles, width=420, bg_color="#f8fafc", border_color="#cbd5e1", header_color="#0f172a"):
    h_style = ParagraphStyle(
        "CardHStyle",
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor(header_color)
    )
    elements = [Paragraph(header_text, h_style), Spacer(1, 6)]
    for p in body_paragraphs:
        elements.append(p)
        elements.append(Spacer(1, 4))
    
    t = Table([[elements]], colWidths=[width])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(bg_color)),
        ('BOX', (0, 0), (-1, -1), 1.0, colors.HexColor(border_color)),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
    ]))
    return t


def build_presentation(output_pdf_path="tcp_rl_presentation.pdf"):
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=PAGE_SIZE,
        leftMargin=40,
        rightMargin=40,
        topMargin=28,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    body_style = ParagraphStyle(
        "BodyDark",
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#334155")
    )
    bullet_style = ParagraphStyle(
        "BulletDark",
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#334155"),
        leftIndent=12,
        firstLineIndent=-12
    )
    bold_style = ParagraphStyle(
        "BoldDark",
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#0f172a")
    )

    story = []

    # =========================================================================
    # SLIDE 1: Title Slide
    # =========================================================================
    story.append(Spacer(1, 45))
    
    # Badge Pill
    badge_style = ParagraphStyle(
        "TitleBadge",
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=12,
        textColor=colors.HexColor("#0284c7")
    )
    story.append(Paragraph("DEEP REINFORCEMENT LEARNING &bull; NETWORK PROTOCOL OPTIMIZATION", badge_style))
    story.append(Spacer(1, 10))

    # Main Title
    big_title_style = ParagraphStyle(
        "BigTitle",
        fontName="Helvetica-Bold",
        fontSize=28,
        leading=33,
        textColor=colors.HexColor("#0f172a")
    )
    story.append(Paragraph("TCP-RL-ns3: Reinforcement Learning-Driven<br/>TCP Congestion Control in ns-3", big_title_style))
    story.append(Spacer(1, 12))

    # Subtitle
    sub_title_style = ParagraphStyle(
        "SubTitle",
        fontName="Helvetica",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#475569")
    )
    story.append(Paragraph("Autonomous Bandwidth-Delay Product (BDP) Tracking & Low-Latency Queue Management under Non-Stationary Bottlenecks", sub_title_style))
    story.append(Spacer(1, 35))

    # Info Grid / Metadata Card
    info_data = [
        [
            Paragraph("<b>Authors:</b> Shubh Shetgaonkar & Abinav Pallathoor", body_style),
            Paragraph("<b>Simulator:</b> ns-3.40 + ns3-gym (ZeroMQ / Protobuf)", body_style),
        ],
        [
            Paragraph("<b>Framework:</b> PyTorch & Stable-Baselines3 (PPO)", body_style),
            Paragraph("<b>Repository:</b> github.com/shetgaonkarshubh/tcp-rl-ns3", body_style),
        ],
        [
            Paragraph("<b>Control Domain:</b> BBR-style Congestion Window Adjustment", body_style),
            Paragraph("<b>License:</b> MIT Open Source License", body_style)
        ]
    ]
    info_table = Table(info_data, colWidths=[430, 440])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
        ('BOX', (0, 0), (-1, -1), 1.2, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 14),
        ('RIGHTPADDING', (0, 0), (-1, -1), 14),
    ]))
    story.append(info_table)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 2: Problem Statement & Motivation
    # =========================================================================
    story.extend(create_slide_header(
        "Background & Challenges",
        "The Congestion Control Dilemma: Bufferbloat vs. Underutilization",
        "Traditional heuristics struggle with modern non-stationary and latency-sensitive networks.",
        styles
    ))

    c1_content = [
        Paragraph("<b>1. Loss-Based Protocols (Reno, Cubic):</b>", bold_style),
        Paragraph("&bull; Assume packet drops are the primary signal of congestion.", bullet_style),
        Paragraph("&bull; Continuously push packets into intermediate router buffers until tail-drop occurs, inducing massive <b>bufferbloat</b> and latency inflation.", bullet_style),
        Paragraph("&bull; Halve cwnd aggressively upon loss, severely degrading throughput on modern cellular/Wi-Fi links with stochastic wireless drops.", bullet_style),
        Spacer(1, 4),
        Paragraph("<b>2. Delay-Based & Model-Based Approaches (BBR, Vegas):</b>", bold_style),
        Paragraph("&bull; Aim to operate at the Kleinrock optimal operating point (maximum throughput, minimum RTT).", bullet_style),
        Paragraph("&bull; Hand-crafted state machines can struggle when bottleneck bandwidth fluctuates unpredictably or when competing with aggressive flows.", bullet_style)
    ]
    card1 = make_card("Limitations of Legacy TCP", c1_content, styles, width=425, bg_color="#fff1f2", border_color="#fecdd3", header_color="#9f1239")

    c2_content = [
        Paragraph("<b>Why Deep Reinforcement Learning (DRL)?</b>", bold_style),
        Paragraph("&bull; <b>Non-Linear State Perception:</b> Directly processes multi-dimensional telemetry (RTT trends, flight sizes, queue inflation) instead of simple threshold logic.", bullet_style),
        Paragraph("&bull; <b>Adaptive Control Policies:</b> Learns asymmetric adjustments—rapidly draining queues on bottleneck constrictions while gently probing during expansions.", bullet_style),
        Spacer(1, 4),
        Paragraph("<b>Project Objectives:</b>", bold_style),
        Paragraph("&bull; Couple <b>ns-3.40</b> with <b>Stable-Baselines3</b> via high-speed IPC (ns3-gym).", bullet_style),
        Paragraph("&bull; Formulate a BBR-style observation and reward schema that targets the dynamic BDP sweet-spot.", bullet_style),
        Paragraph("&bull; Train a resilient PPO agent robust to severe, sudden bandwidth shifts.", bullet_style)
    ]
    card2 = make_card("The Deep RL Opportunity", c2_content, styles, width=425, bg_color="#f0fdf4", border_color="#bbf7d0", header_color="#166534")

    two_col = Table([[card1, card2]], colWidths=[435, 435])
    two_col.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(two_col)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 3: System Architecture
    # =========================================================================
    story.extend(create_slide_header(
        "System Architecture",
        "Co-Simulation Pipeline: ns-3 Simulator & Python RL Agent",
        "A bi-directional, synchronous IPC architecture bridging C++ discrete-event simulation with PyTorch.",
        styles
    ))

    arch_data = [
        [
            Paragraph("<b>Layer</b>", bold_style),
            Paragraph("<b>Component</b>", bold_style),
            Paragraph("<b>Key Responsibilities & Data Exchange</b>", bold_style)
        ],
        [
            Paragraph("<b>Simulation<br/>(C++17)</b>", body_style),
            Paragraph("<b>ns-3.40 Network Core</b><br/><code>rl-tcp.cc</code>", body_style),
            Paragraph("&bull; Simulates realistic point-to-point network topologies with bottleneck links.<br/>&bull; Hooks into <code>TcpSocketState</code> to intercept cwnd, ssThresh, RTT, segments in flight, and congestion states (CA_OPEN, CA_RECOVERY, etc.).<br/>&bull; Executes commanded cwnd and ssThresh updates at each 100 ms control interval.", body_style)
        ],
        [
            Paragraph("<b>IPC Bridge</b>", body_style),
            Paragraph("<b>ns3-gym Module</b><br/>ZeroMQ / Protobuf", body_style),
            Paragraph("&bull; Synchronous IPC socket over localhost port <code>7144</code>.<br/>&bull; Serializes raw C++ telemetry into Protocol Buffer messages and returns agent decisions back to the simulator with zero disk overhead.", body_style)
        ],
        [
            Paragraph("<b>Gymnasium<br/>Environment</b>", body_style),
            Paragraph("<b>CustomTcpEnv</b><br/><code>tcp_env.py</code>", body_style),
            Paragraph("&bull; Emulates dynamic network environments with a stochastic Markov bottleneck engine.<br/>&bull; Calibrates physical Base RTT (84 ms) and derives time-invariant 8-dim observations.<br/>&bull; Computes multi-objective reward balancing BDP tracking, throughput, and delay penalty.", body_style)
        ],
        [
            Paragraph("<b>Learning<br/>Agent</b>", body_style),
            Paragraph("<b>PPO / PyTorch</b><br/><code>train.py</code>", body_style),
            Paragraph("&bull; Stable-Baselines3 PPO with customized MLP policy (SiLU activations).<br/>&bull; Maintains running reward normalization via <code>VecNormalize</code>.", body_style)
        ]
    ]

    arch_table = Table(arch_data, colWidths=[90, 160, 620])
    arch_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BOX', (0, 0), (-1, -1), 1.0, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP')
    ]))
    story.append(arch_table)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 4: Stochastic Bottleneck Markov Engine
    # =========================================================================
    story.extend(create_slide_header(
        "Environment Modeling",
        "Stochastic Bottleneck Markov Engine & Queue Dynamics",
        "Realistic non-stationary evaluation forcing the agent to track unpredictable capacity jumps.",
        styles
    ))

    sb1_content = [
        Paragraph("<b>1. Random Capacity Transitions:</b>", bold_style),
        Paragraph("&bull; Real-world links (5G, satellite, multi-tenant WAN) feature abrupt bandwidth variations.", bullet_style),
        Paragraph("&bull; The environment randomly samples target BDP levels uniformly in <b>[15.0, 36.5] MSS</b>.", bullet_style),
        Paragraph("&bull; Enforces severe shifts: <b>|&Delta; BDP| &ge; 6.5 MSS</b> per jump to prevent trivial steady-state overfitting.", bullet_style),
        Spacer(1, 4),
        Paragraph("<b>2. Stochastic Hold Durations:</b>", bold_style),
        Paragraph("&bull; Bottleneck capacity persists for random hold intervals of <b>45 to 85 steps (4.5s to 8.5s)</b>.", bullet_style),
        Paragraph("&bull; Forces the policy to balance immediate queue draining during constrictions with stable bandwidth holding.", bullet_style)
    ]
    sb_card1 = make_card("Markov Bottleneck Engine", sb1_content, styles, width=425, bg_color="#f8fafc", border_color="#cbd5e1")

    sb2_content = [
        Paragraph("<b>3. Adaptive Base RTT Calibration:</b>", bold_style),
        Paragraph("&bull; Continuously tracks lowest observed RTT to calibrate the physical propagation delay: <code>base_rtt &approx; 84.0 ms</code>.", bullet_style),
        Paragraph("&bull; Uses a 3.5% safety margin to prevent misclassifying small queuing ripples as propagation changes.", bullet_style),
        Spacer(1, 4),
        Paragraph("<b>4. Physical Queue Emulation:</b>", bold_style),
        Paragraph("&bull; When link capacity drops while cwnd remains high, queue buildup is simulated: excess packets induce instantaneous RTT inflation.", bullet_style),
        Paragraph("&bull; Queue ceiling set at <b>+25% above Base RTT (~105 ms)</b>, matching the classic BBR bufferbloat boundary.", bullet_style)
    ]
    sb_card2 = make_card("Delay & Queue Dynamics", sb2_content, styles, width=425, bg_color="#f8fafc", border_color="#cbd5e1")

    sb_col = Table([[sb_card1, sb_card2]], colWidths=[435, 435])
    sb_col.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(sb_col)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 5: RL Formulation
    # =========================================================================
    story.extend(create_slide_header(
        "Reinforcement Learning Formulation",
        "State Space, Action Space & Multi-Objective Reward Function",
        "Engineered for rapid convergence, sample efficiency, and low-latency queuing discipline.",
        styles
    ))

    # Three mini cards across
    obs_content = [
        Paragraph("<b>8 Normalized Features [-5, 5]:</b>", bold_style),
        Paragraph("1. <code>(cwnd / BDP) - 1.0</code> (BDP error)", bullet_style),
        Paragraph("2. <code>log(cwnd / segment_size)</code>", bullet_style),
        Paragraph("3. <code>bytes_in_flight / cwnd</code>", bullet_style),
        Paragraph("4. <code>(RTT - BaseRTT) / BaseRTT</code> (Inflation)", bullet_style),
        Paragraph("5. <code>&Delta; RTT / step</code> (Gradient)", bullet_style),
        Paragraph("6. TCP state enum (OPEN, CWR, LOSS)", bullet_style),
        Paragraph("7. Segment loss flag", bullet_style),
        Paragraph("8. Normalized Target BDP feature", bullet_style)
    ]
    card_obs = make_card("Observation Space (8D)", obs_content, styles, width=280, bg_color="#f8fafc", border_color="#cbd5e1")

    act_content = [
        Paragraph("<b>7 Discrete Deltas (&Delta; cwnd):</b>", bold_style),
        Paragraph("<b>0:</b> <code>-4.0 MSS</code> (Emergency drain)", bullet_style),
        Paragraph("<b>1:</b> <code>-2.0 MSS</code> (Moderate backoff)", bullet_style),
        Paragraph("<b>2:</b> <code>-0.5 MSS</code> (Fine trim down)", bullet_style),
        Paragraph("<b>3:</b> <code> 0.0 MSS</code> (BDP Sweet-Spot Hold)", bullet_style),
        Paragraph("<b>4:</b> <code>+0.5 MSS</code> (Fine trim up)", bullet_style),
        Paragraph("<b>5:</b> <code>+2.0 MSS</code> (Bandwidth probe)", bullet_style),
        Paragraph("<b>6:</b> <code>+4.0 MSS</code> (Aggressive ramp-up)", bullet_style),
        Spacer(1, 4),
        Paragraph("<i>Applied every 100 ms control step.</i>", body_style)
    ]
    card_act = make_card("Action Space (7 Actions)", act_content, styles, width=280, bg_color="#f8fafc", border_color="#cbd5e1")

    rew_content = [
        Paragraph("<b>Multi-Term Reward:</b>", bold_style),
        Paragraph("<code>R = R_bdp + R_prog - C_over - C_delay - C_loss</code>", body_style),
        Spacer(1, 4),
        Paragraph("&bull; <b>R_bdp:</b> High reward when within &plusmn;0.6 MSS of BDP.", bullet_style),
        Paragraph("&bull; <b>R_prog:</b> Throughput progress bonus for timely ACKs.", bullet_style),
        Paragraph("&bull; <b>C_delay:</b> <code>4.0 * (excess_delay)^1.5</code> penalizing bufferbloat.", bullet_style),
        Paragraph("&bull; <b>C_over:</b> Quadratic penalty for expanding cwnd above bottleneck BDP.", bullet_style),
        Paragraph("&bull; <b>C_loss:</b> -4.5 penalty upon TCP RECOVERY or LOSS.", bullet_style)
    ]
    card_rew = make_card("Reward Formulation", rew_content, styles, width=280, bg_color="#f8fafc", border_color="#cbd5e1")

    three_col = Table([[card_obs, card_act, card_rew]], colWidths=[290, 290, 290])
    three_col.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(three_col)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 6: Training Pipeline & Hyperparameters
    # =========================================================================
    story.extend(create_slide_header(
        "Agent Training & Hyperparameters",
        "PPO Configuration & 3-Pane Automated Orchestration",
        "End-to-end synchronized training pipeline featuring tmux process isolation and live telemetry logging.",
        styles
    ))

    hp_data = [
        [Paragraph("<b>Hyperparameter</b>", bold_style), Paragraph("<b>Value</b>", bold_style), Paragraph("<b>Rationale</b>", bold_style)],
        [Paragraph("Algorithm", body_style), Paragraph("PPO (Proximal Policy Optimization)", body_style), Paragraph("Sample-efficient on-policy learning with stable clipping bounds.", body_style)],
        [Paragraph("Policy Network", body_style), Paragraph("MLP [128, 128] Actor-Critic", body_style), Paragraph("Compact representation capable of sub-millisecond inference.", body_style)],
        [Paragraph("Activation Function", body_style), Paragraph("SiLU (Swish)", body_style), Paragraph("Smooth gradient propagation for continuous control boundaries.", body_style)],
        [Paragraph("Learning Rate", body_style), Paragraph("Linear Decay: 3e-4 &rarr; 3e-5", body_style), Paragraph("Allows broad initial exploration, annealing to precise steady-state hold.", body_style)],
        [Paragraph("Rollout Horizon", body_style), Paragraph("1,024 steps per update", body_style), Paragraph("Spans multiple random bottleneck shift cycles for balanced variance.", body_style)],
        [Paragraph("Batch Size / Epochs", body_style), Paragraph("128 / 5 epochs per rollout", body_style), Paragraph("Prevents catastrophic forgetting while maximizing sample reuse.", body_style)],
        [Paragraph("Discount Factor (&gamma;)", body_style), Paragraph("0.98", body_style), Paragraph("Balances immediate queue penalties with mid-horizon throughput.", body_style)],
        [Paragraph("Reward Normalization", body_style), Paragraph("VecNormalize (clip=10.0)", body_style), Paragraph("Stabilizes value function learning across shifting reward regimes.", body_style)],
    ]
    hp_table = Table(hp_data, colWidths=[150, 190, 530])
    hp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BOX', (0, 0), (-1, -1), 1.0, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0, 0), (-1, -1), 4.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ]))
    story.append(hp_table)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 7: Empirical Results
    # =========================================================================
    story.extend(create_slide_header(
        "Empirical Benchmark & Evaluation",
        "Evaluation Dynamics under Severe Bottleneck Capacity Fluctuations",
        "Deterministic evaluation across 400 steps demonstrating tight BDP tracking and zero bufferbloat.",
        styles
    ))

    img_path = "/home/shubh/tcpai/rl_tcp_zoomed_performance.png"
    img_element = None
    if os.path.exists(img_path):
        # 4:3 aspect ratio display inside slide: 470 x 360
        img_element = Image(img_path, width=480, height=365)

    results_content = [
        Paragraph("<b>Key Empirical Findings:</b>", bold_style),
        Spacer(1, 4),
        Paragraph("<b>1. BDP Tracking Accuracy (MAE = 0.65 MSS):</b>", bold_style),
        Paragraph("&bull; The agent holds cwnd strictly inside the <b>&plusmn;1.5 MSS BBR sweet-spot band</b> (shaded region in Subplot 1).", bullet_style),
        Paragraph("&bull; When the bottleneck drops abruptly from 36.5 to 15.0 MSS, the agent executes asymmetric -4.0 MSS actions to rapidly drain the pipe.", bullet_style),
        Spacer(1, 6),
        Paragraph("<b>2. Strict Latency & Queue Control:</b>", bold_style),
        Paragraph("&bull; Observed RTT hovers tightly around physical Base RTT (<b>84.0 ms</b>).", bullet_style),
        Paragraph("&bull; 95th-percentile RTT is bounded well below the +25% BBR queue threshold, confirming total elimination of bufferbloat.", bullet_style),
        Spacer(1, 6),
        Paragraph("<b>3. Sustained Positive Reward:</b>", bold_style),
        Paragraph("&bull; Rolling 15-step mean reward remains positive throughout evaluation, recovering within 2 steps post-shift.", bullet_style)
    ]
    results_card = make_card("Performance Analysis", results_content, styles, width=370, bg_color="#f8fafc", border_color="#cbd5e1")

    res_col = Table([[img_element, results_card]], colWidths=[495, 375])
    res_col.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(res_col)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 8: Getting Started & Future Directions
    # =========================================================================
    story.extend(create_slide_header(
        "Reproduction & Future Work",
        "Getting Started & Research Roadmap",
        "Complete standalone reproduction workflow and upcoming enhancements.",
        styles
    ))

    c_run_content = [
        Paragraph("<b>Quick Start Guide:</b>", bold_style),
        Spacer(1, 4),
        Paragraph("<b>1. Compile ns-3 Simulation Binary:</b>", bold_style),
        Paragraph("<code>cp rl-tcp.cc ~/ns-allinone-3.40/ns-3.40/contrib/opengym/examples/rl-tcp/</code>", body_style),
        Paragraph("<code>cd ~/ns-allinone-3.40/ns-3.40 && ./ns3 build</code>", body_style),
        Spacer(1, 6),
        Paragraph("<b>2. Automated 3-Pane Launch (tmux):</b>", bold_style),
        Paragraph("<code>cd ~/tcpai && ./run_pipeline_tmux.sh</code>", body_style),
        Paragraph("&bull; Pane 0: Starts ns-3 server on port 7144.<br/>&bull; Pane 1: Trains PPO policy with live telemetry.<br/>&bull; Pane 2: Automatically plots <code>rl_tcp_zoomed_performance.png</code> upon trace generation.", bullet_style),
        Spacer(1, 6),
        Paragraph("<b>3. Standalone Evaluation & Real-Time Demo:</b>", bold_style),
        Paragraph("<code>python plot_results.py</code>", body_style),
        Paragraph("<code>./run_realtime_demo.sh  # Live HUD Dashboard</code>", body_style)
    ]
    card_run = make_card("Execution & Reproduction", c_run_content, styles, width=425, bg_color="#f8fafc", border_color="#cbd5e1")

    c_future_content = [
        Paragraph("<b>Future Research Horizons:</b>", bold_style),
        Spacer(1, 4),
        Paragraph("<b>1. Multi-Flow Fairness & Competition:</b>", bold_style),
        Paragraph("&bull; Evaluate PPO agent behavior when sharing bottleneck links with aggressive Cubic and BBRv2 cross-traffic.", bullet_style),
        Spacer(1, 6),
        Paragraph("<b>2. Variable Delay Environments (LEO Satellites):</b>", bold_style),
        Paragraph("&bull; Extend state features to handle time-varying propagation delays characteristic of Starlink / LEO constellations.", bullet_style),
        Spacer(1, 6),
        Paragraph("<b>3. Linux Kernel eBPF Export:</b>", bold_style),
        Paragraph("&bull; Compile trained MLP actor network to eBPF bytecode for native in-kernel TCP congestion ops in Linux.", bullet_style),
        Spacer(1, 6),
        Paragraph("<b>4. Continuous Action Spaces (PPO/SAC):</b>", bold_style),
        Paragraph("&bull; Benchmark discrete delta actions against continuous pacing rate multipliers.", bullet_style)
    ]
    card_future = make_card("Future Directions", c_future_content, styles, width=425, bg_color="#f0f9ff", border_color="#bae6fd", header_color="#0369a1")

    future_col = Table([[card_run, card_future]], colWidths=[435, 435])
    future_col.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(future_col)

    # Build the document
    doc.build(story, canvasmaker=SlideCanvas)
    print(f"[*] Presentation PDF generated successfully: {output_pdf_path}")

if __name__ == "__main__":
    build_presentation("/home/shubh/tcpai/tcp_rl_presentation.pdf")
