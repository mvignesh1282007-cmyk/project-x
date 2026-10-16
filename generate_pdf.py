"""
generate_pdf.py
Generates a comprehensive, professional PDF guide for:
"10 Real-Time AI Software Project Topics in Manufacturing"
Tailored for:
- High judge understanding & engagement
- Easy data acquisition (Webcam, Kaggle 1-click datasets, or simple CSVs)
- Laptop-only execution with zero specialized hardware
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Canvas that enables two-pass page numbering and clean running headers/footers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))
        
        # Don't draw headers/footers on cover page
        if self._pageNumber > 1:
            # Header
            self.drawString(54, letter[1] - 36, "AI in Manufacturing — 10 Real-Time Software Project Topics (Judge-Ready)")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.5)
            self.line(54, letter[1] - 42, letter[0] - 54, letter[1] - 42)
            
            # Footer
            page_text = f"Page {self._pageNumber} of {page_count}"
            self.drawRightString(letter[0] - 54, 32, page_text)
            self.drawString(54, 32, "Confidential & Educational • Built for Laptop Prototyping & Live Demonstrations")
            self.line(54, 44, letter[0] - 54, 44)
            
        self.restoreState()

def build_pdf(filename="AI_in_Manufacturing_10_RealTime_Projects.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Palette
    PRIMARY = colors.HexColor("#1A365D")   # Deep Industrial Navy
    SECONDARY = colors.HexColor("#2B6CB0") # Vibrant Blue
    DARK_TEXT = colors.HexColor("#2D3748") # Charcoal
    LIGHT_BG = colors.HexColor("#F7FAFC")  # Off-white / light slate
    BORDER_COL = colors.HexColor("#E2E8F0")
    ACCENT_GREEN = colors.HexColor("#22543D")
    
    # Custom Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=PRIMARY,
        spaceAfter=8
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=SECONDARY,
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=PRIMARY,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'ProjectHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=SECONDARY,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'StandardBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=DARK_TEXT,
        spaceAfter=6
    )
    
    bullet_style = ParagraphStyle(
        'CustomBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=DARK_TEXT,
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=3
    )

    pitch_style = ParagraphStyle(
        'PitchText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#2C5282")
    )
    
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=DARK_TEXT
    )
    
    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.white
    )

    story = []

    # ================= COVER / TITLE SECTION =================
    story.append(Paragraph("10 Real-Time Software Projects in AI for Manufacturing", title_style))
    story.append(Paragraph("A Comprehensive Blueprint for High-Scoring Judge Demonstrations, Easy Data Acquisition & Laptop-Only Prototyping", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=SECONDARY, spaceBefore=0, spaceAfter=14))
    
    # Executive Summary Card
    exec_summary_html = """
    <b>Executive Overview:</b> When presenting an Artificial Intelligence project in Manufacturing to competition judges or academic evaluators, two pitfalls frequently ruin projects: (1) picking abstract physics/sensor models that judges cannot visualize in 30 seconds, or (2) relying on industrial datasets that are difficult to gather without expensive factory hardware.<br/><br/>
    This guide presents <b>10 carefully curated real-time software projects</b> designed specifically to run entirely on a <b>standard laptop</b>, utilize <b>freely accessible datasets or laptop webcams</b>, and feature <b>interactive, visual live dashboards</b> that judges immediately understand and appreciate.
    """
    
    summary_table = Table([[Paragraph(exec_summary_html, body_style)]], colWidths=[letter[0] - 108])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EBF8FF")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#BEE3F8")),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 12))

    # Judge Scoring Rubric Guide
    story.append(Paragraph("What Judges Look For in a Winning Manufacturing AI Project", h1_style))
    story.append(Paragraph("Judges evaluate projects across four core industrial pillars:", body_style))
    story.append(Paragraph("• <b>Instant Visual Comprehension:</b> The judge must understand the problem and solution in 10 seconds (e.g., 'Green = Quality Passed', 'Red = Scratch Defect Detected').", bullet_style))
    story.append(Paragraph("• <b>Live Real-Time Interactivity:</b> The system must not be a static script. Judges want to click a button, inject an error, or adjust a slider and watch the AI react live.", bullet_style))
    story.append(Paragraph("• <b>Clear Business & Operational ROI:</b> Explaining practical metrics like scrap rate reduction, uptime hours saved, or safety compliance.", bullet_style))
    story.append(Paragraph("• <b>Self-Contained Feasibility:</b> Running 100% on a laptop using free tools (Python, OpenCV, Scikit-learn, Streamlit) without requiring industrial machinery.", bullet_style))
    
    story.append(Spacer(1, 10))

    # ================= THE 10 PROJECTS DETAILED =================
    story.append(HRFlowable(width="100%", thickness=1, color=BORDER_COL, spaceBefore=8, spaceAfter=14))
    story.append(Paragraph("The 10 Judge-Ready Manufacturing AI Project Blueprints", h1_style))
    
    projects = [
        {
            "num": "01",
            "title": "Factory PPE & Industrial Safety Compliance Monitor",
            "pitch": "Real-time edge vision monitoring that verifies shop-floor workers wear required safety gear (helmets, high-vis vests, protective eyewear) before entering active heavy machinery zones.",
            "problem": "Factory floor accidents cost manufacturing companies billions annually in insurance, worker downtime, and legal liability when workers bypass safety equipment rules.",
            "data_source": "<b>Super Easy:</b> Use your laptop's built-in webcam! You can test it in real time by wearing a baseball cap/hat or bright shirt. Free public datasets include <i>Hard Hat and Safety Vest Detection</i> on Kaggle or Roboflow (pre-labeled with helmet/vest classes).",
            "ai_models": "Ultralytics YOLOv8n (nano) fine-tuned for PPE or pre-trained safety gear weights. Extremely fast on laptop CPU.",
            "tools": "Python, OpenCV, Ultralytics YOLOv8, Streamlit, pyttsx3 (for audio alarm voice warning).",
            "demo_action": "Have the judge or yourself stand in front of the laptop webcam. Without a cap/hat: The system flashes RED: 'SAFETY VIOLATION: NO HARD HAT DETECTED' and logs a violation ticket. Put on a cap or safety gear: Instant GREEN: 'COMPLIANT: WORKER AUTHORIZED FOR SHOP FLOOR'.",
            "metrics": "Real-time Compliance Score (%), Active Violations Count, Zone Clearance Status, Hourly Audit Log."
        },
        {
            "num": "02",
            "title": "Assembly Line Bottleneck & Production Delay Predictor",
            "pitch": "A smart production line tracking platform that monitors multi-station manufacturing flow (Stamping -> Welding -> Painting -> Assembly) to predict bottlenecks and shipment delays 2 hours in advance.",
            "problem": "When one station slows down due to minor part shortages or operator pauses, upstream inventory piles up, halting the entire multi-million dollar production line.",
            "data_source": "<b>Super Easy:</b> Simple tabular CSV data containing 5 clean columns: [Timestamp, Station_ID, Units_In, Units_Out, Cycle_Time_Sec, Buffer_Queue]. Can be simulated via 20 lines of Python or downloaded from Kaggle OEE / Manufacturing Flow datasets.",
            "ai_models": "Random Forest Regressor / XGBoost to predict queue delay times; moving average anomaly detection to identify stalled stations.",
            "tools": "Python, Pandas, Scikit-learn, Streamlit, Plotly.",
            "demo_action": "Display an interactive visual factory flow diagram with 4 stations. Provide an interactive slider: 'Stamping Station Speed'. Slide the speed down: The judge watches the Station 2 icon turn bright RED, queue buffer fill up, and the AI alert: 'Predicted Packaging Delay: +48 mins. Recommended: Reroute 25% throughput to Line B'.",
            "metrics": "Line OEE (Overall Equipment Effectiveness), Bottleneck Risk Factor, Predicted Daily Output vs Target, Buffer Health."
        },
        {
            "num": "03",
            "title": "Real-Time Machine Health & Predictive Maintenance Platform",
            "pitch": "Monitors industrial equipment telemetry (vibration, temperature, RPM, torque) to predict equipment failure hours before breakdown, eliminating unplanned factory downtime.",
            "problem": "Unplanned machine breakdown in automotive and CNC manufacturing costs an average of $22,000 per minute in halted assembly line labor and emergency repair parts.",
            "data_source": "<b>Super Easy:</b> The official UCI AI4I 2020 Predictive Maintenance Dataset (10,000 verified manufacturing operating records with temperature, torque, RPM, tool wear, and failure modes). Downloads in 2 seconds.",
            "ai_models": "XGBoost Classifier (Failure probability & Health score) + Random Forest (Multi-class failure mode diagnosis: Thermal, Power, Tool Wear, Overstrain) + Isolation Forest (unsupervised drift).",
            "tools": "Python, Scikit-learn, XGBoost, Streamlit, Plotly.",
            "demo_action": "Show a live ticking telemetry stream. Provide a 'Judge Fault Injection Panel' with buttons: 'Inject Overheating Fault' or 'Inject Spindle Overstrain'. Click the button: Health gauge instantly plummets from 95% to 22%, turns red, and prints an autonomous work-order dispatch ticket for the technician.",
            "metrics": "Machine Health Index (0-100%), Est. Remaining Useful Life (RUL), Failure Mode Diagnosis, Unplanned Cost Prevented ($)."
        },
        {
            "num": "04",
            "title": "Factory Energy Spike & Carbon Cost Forecaster",
            "pitch": "Monitors electrical draw across high-power factory machinery in real time, predicting peak-tariff demand spikes and detecting phantom energy waste during idle shifts.",
            "problem": "Manufacturing plants face severe utility penalties when factory power draw crosses peak-demand utility thresholds, and idle machinery wastes up to 30% of total plant electricity.",
            "data_source": "<b>Super Easy:</b> Kaggle <i>Steel Industry Energy Consumption Dataset</i> (clean tabular time-series tracking Active Power, Reactive Power, Power Factor, and CO2 emissions) or a simulated smart-meter power stream.",
            "ai_models": "XGBoost Regressor or Random Forest for 24-hour demand forecasting; Isolation Forest for detecting machinery left running while idle.",
            "tools": "Python, Pandas, Scikit-learn, Streamlit, Plotly.",
            "demo_action": "Live real-time energy strip chart with a red 'Peak Tariff Penalty Zone' line. Toggle machine states (Press On/Off, Furnace On/Off). Show AI forecasting: 'Warning: 3:00 PM Spindle Surge will breach 450 kW peak threshold. Recommended: Stagger Furnace cycle by 20 minutes to save $3,200/month'.",
            "metrics": "Real-time kW Demand, Predicted Peak-Hour Surge Risk, Daily Carbon Footprint (kg CO2), Est. Monthly Tariff Penalty."
        },
        {
            "num": "05",
            "title": "Raw Material Stockout & Autonomous Reorder Forecaster",
            "pitch": "Predictive inventory platform that forecasts raw material depletion rates and alerts procurement before critical manufacturing supplies (sheet metal, resins, fasteners) run dry.",
            "problem": "Stockouts of even tiny raw materials (e.g., specialized bolts) halt entire multi-million dollar product assemblies, while excessive over-ordering ties up capital in warehouse storage.",
            "data_source": "<b>Super Easy:</b> A simple tabular CSV tracking inventory levels, daily consumption, and supplier lead times. Free Kaggle <i>Supply Chain Logistics / Inventory Dataset</i> or a generated 50-row CSV file.",
            "ai_models": "Linear/Ridge Regression, Exponential Smoothing, or Random Forest to forecast demand based on historical seasonal trends.",
            "tools": "Python, Pandas, Scikit-learn, Streamlit, SQLite.",
            "demo_action": "Inventory dashboard showing visual stock meters for 5 critical materials. Click 'Simulate Rush Order of 1,000 Units': Raw aluminum stock plummets into the red zone. The AI immediately generates an automated Purchase Requisition: 'Vendor: AlumCorp | Reorder Qty: 450 kg | ETA: 48h to prevent line shutdown'.",
            "metrics": "Days of Supply Remaining, Reorder Point (ROP), Safety Stock Alert, Stockout Risk Probability (%)."
        },
        {
            "num": "06",
            "title": "Packaging Completeness & Component Verifier",
            "pitch": "Automated vision quality gate at the final packing station that verifies every boxed customer shipment contains all required components (main unit, cable, manual, hardware kit) before carton sealing.",
            "problem": "Missing items inside sealed retail or industrial product boxes lead to high customer returns, negative brand reviews, and expensive individual shipping of forgotten parts.",
            "data_source": "<b>Super Easy:</b> Use your laptop webcam! Place 3-4 ordinary items on your desk (e.g., smartphone, charging cable, pen, notebook) inside a shallow cardboard box or marked tray. Pre-trained models require zero external dataset downloads.",
            "ai_models": "YOLOv8-nano (pre-trained on COCO dataset, recognizes 80 common object classes out of the box with zero custom training).",
            "tools": "Python, OpenCV, Ultralytics YOLOv8, Streamlit.",
            "demo_action": "Place the box in view of your webcam. When all items are inside: Dashboard shows 'BOX COMPLETE: 4/4 ITEMS DETECTED - APPROVED FOR TAPE SEAL'. Remove the cable/pen: The AI instantly flashes 'PACKAGING ERROR: MISSING CHARGING CABLE - SHIPMENT HALTED'.",
            "metrics": "Completeness Checklist, Missing Item Identifier, Packing Verification Cycle Time, Shipment Audit Trail."
        },
        {
            "num": "07",
            "title": "Machine Operator Fatigue & Workplace Attention Guard",
            "pitch": "An AI vision safety guard that tracks heavy machinery (press/saw/lathe) operators for signs of micro-sleep, eye closure, or mobile phone distraction, triggering emergency spindle locks.",
            "problem": "Fatigued or distracted operators working lethal stamping presses and high-speed cutting saws cause thousands of amputations and severe injuries annually.",
            "data_source": "<b>Super Easy:</b> Your laptop's built-in webcam. Uses real-time facial landmark detection (MediaPipe Face Mesh) to calculate the Eye Aspect Ratio (EAR) and head pose angle. Zero dataset downloading required.",
            "ai_models": "MediaPipe Face Mesh (468 landmarks) + Eye Aspect Ratio (EAR) metric + Random Forest / Threshold Classifier.",
            "tools": "Python, OpenCV, MediaPipe, Streamlit.",
            "demo_action": "The presenter sits in front of the laptop. Close eyes for more than 2 seconds: Screen triggers a flashing red warning with loud buzzer audio: 'EMERGENCY: OPERATOR DROWSINESS DETECTED - SPINDLE SAFETY BRAKE ENGAGED'. Open eyes: Returns to 'OPERATOR ALERT & ENGAGED'.",
            "metrics": "Attention Index (0-100%), Blink Duration (ms), Fatigue Warning Event Log, Safety Interlock Status."
        },
        {
            "num": "08",
            "title": "Factory Production Demand & Batch Size Optimizer",
            "pitch": "Forecasts customer market demand 30 days ahead and calculates the optimal factory production batch size to minimize both setup downtime and warehouse inventory holding costs.",
            "problem": "Overproduction fills warehouses with unsold obsolete stock, while underproduction causes backorders and lost market share.",
            "data_source": "<b>Super Easy:</b> Kaggle <i>Product Demand Forecasting Dataset</i> (historical sales orders across multiple manufacturing product categories) or simple monthly demand time-series CSV.",
            "ai_models": "Prophet / ARIMA or XGBoost Regressor for time-series demand; Economic Order Quantity (EOQ) optimization algorithm.",
            "tools": "Python, Pandas, Scikit-learn, Streamlit, Plotly.",
            "demo_action": "Dashboard showing historic demand curves and 30-day forecast cones. Provide a slider: 'Simulate Raw Material Price Increase (+15%)' or 'Simulate Demand Spike (+30%)'. The AI instantly recalculates recommended batch sizes, showing the trade-off curve between machine changeover cost and inventory storage cost.",
            "metrics": "30-Day Demand Forecast, Recommended Economic Batch Size, Warehouse Capacity Utilization, Est. Holding Cost ($)."
        },
        {
            "num": "09",
            "title": "Smart Factory Cell Digital Twin & OEE Live Monitor",
            "pitch": "A real-time software digital twin of a 3-station manufacturing cell that mirrors telemetry, tracks live OEE (Availability, Performance, Quality), and predicts cell throughput.",
            "problem": "Factory managers struggle with fragmented data across isolated machines, making it difficult to understand true shop-floor efficiency in real time.",
            "data_source": "<b>Super Easy:</b> Synthetic digital twin telemetry generator (or Kaggle Industrial Robot / Sensor log). Generates realistic synchronized machine cycles with realistic cycle jitter.",
            "ai_models": "Multi-output Regression for cycle time prediction + Isolation Forest for detecting cross-station synchronization lag.",
            "tools": "Python, Streamlit, Plotly, Pandas.",
            "demo_action": "Interactive 2D schematic of a smart factory cell with 3 machines. Live gauges display Availability (92%), Performance (88%), and Quality (98%) -> Total OEE = 79.4% (World Class = 85%). Click 'Trigger Tool Jam on Station 2': The 2D twin updates Station 2 to red, OEE drops live, and AI predicts loss of 42 finished parts.",
            "metrics": "Live OEE Score, Availability / Performance / Quality Breakdown, Cell Throughput Rate (units/hr), Anomaly Status."
        },
        {
            "num": "10",
            "title": "Production Scrap Rate & Process Parameter Optimizer",
            "pitch": "Predicts scrap/waste rates in manufacturing processes (like plastic injection molding or metal stamping) based on operational settings (temperature, pressure, cycle speed), recommending optimal settings to minimize material waste.",
            "problem": "Sub-optimal machine operating parameters create thousands of defect scrap parts every shift, inflating raw material costs and squandering energy.",
            "data_source": "<b>Super Easy:</b> Kaggle <i>Injection Molding Process Dataset</i> or a simple 5-column CSV: [Mold_Temp_C, Injection_Pressure_Bar, Cooling_Time_Sec, Clamping_Force_kN, Scrap_Rate_Pct]. Easily generated or downloaded.",
            "ai_models": "Random Forest Regressor / XGBoost to predict Scrap Rate % + Grid Search / Gradient Descent parameter optimizer.",
            "tools": "Python, Pandas, Scikit-learn, Streamlit, Plotly.",
            "demo_action": "Interactive sliders for machine controls (Pressure: 130 bar, Temp: 195°C). Drag the pressure slider: Scrap Rate gauge jumps into the RED (13.5% Scrap: Flash Defect). Click 'AI Auto-Tune': The AI recommends: 'Reduce pressure by 16 bar and increase cooling by 2.0s to restore 0.8% zero-defect scrap rate'.",
            "metrics": "Predicted Scrap Rate (%), Material Waste Cost ($/hr), Optimal Parameter Setpoints, Quality Loss Index."
        }
    ]

    for p in projects:
        card_content = []
        # Header line with badge
        card_content.append(Paragraph(f"<b>PROJECT {p['num']}: {p['title']}</b>", h2_style))
        
        # Pitch in callout box
        pitch_box = Table([[Paragraph(f"<b>15-Second Judge Pitch:</b> {p['pitch']}", pitch_style)]], colWidths=[letter[0] - 120])
        pitch_box.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EDF2F7")),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]))
        card_content.append(pitch_box)
        card_content.append(Spacer(1, 4))
        
        # Structured Bullet Points
        card_content.append(Paragraph(f"• <b>Industrial Problem & Value:</b> {p['problem']}", bullet_style))
        card_content.append(Paragraph(f"• <b>Where to Get Data (Super Easy):</b> {p['data_source']}", bullet_style))
        card_content.append(Paragraph(f"• <b>AI / ML Models:</b> {p['ai_models']}", bullet_style))
        card_content.append(Paragraph(f"• <b>Tech Stack:</b> {p['tools']}", bullet_style))
        card_content.append(Paragraph(f"• <b>Live Judge Demo Action:</b> {p['demo_action']}", bullet_style))
        card_content.append(Paragraph(f"• <b>Key Dashboard Metrics:</b> {p['metrics']}", bullet_style))
        card_content.append(Spacer(1, 8))
        
        # Wrap project in KeepTogether so each project stays cleanly formatted
        story.append(KeepTogether(card_content))

    # ================= SUMMARY COMPARISON MATRIX =================
    story.append(PageBreak())
    story.append(Paragraph("Quick Comparison Matrix (Select Your Best Match)", h1_style))
    story.append(Paragraph("Review all 10 projects at a glance to select the one that fits your comfort level and presentation preference:", body_style))
    story.append(Spacer(1, 6))

    matrix_data = [
        [
            Paragraph("<b>#</b>", table_header_style),
            Paragraph("<b>Project Title</b>", table_header_style),
            Paragraph("<b>Data Source</b>", table_header_style),
            Paragraph("<b>AI Technique</b>", table_header_style),
            Paragraph("<b>Difficulty</b>", table_header_style),
            Paragraph("<b>Judge Impact</b>", table_header_style)
        ],
        [Paragraph("1", table_cell_style), Paragraph("PPE Safety Monitor", table_cell_style), Paragraph("Laptop Webcam (Real-time)", table_cell_style), Paragraph("YOLOv8-nano Vision", table_cell_style), Paragraph("Medium", table_cell_style), Paragraph("⭐⭐⭐⭐⭐ (Highest)", table_cell_style)],
        [Paragraph("2", table_cell_style), Paragraph("Assembly Line Bottleneck", table_cell_style), Paragraph("Simple Tabular CSV", table_cell_style), Paragraph("XGBoost / Random Forest", table_cell_style), Paragraph("Easy", table_cell_style), Paragraph("⭐⭐⭐⭐ (Operations)", table_cell_style)],
        [Paragraph("3", table_cell_style), Paragraph("Predictive Maintenance", table_cell_style), Paragraph("UCI AI4I 2020 Dataset", table_cell_style), Paragraph("XGBoost + Isolation Forest", table_cell_style), Paragraph("Easy-Medium", table_cell_style), Paragraph("⭐⭐⭐⭐⭐ (Technical)", table_cell_style)],
        [Paragraph("4", table_cell_style), Paragraph("Factory Energy Spikes", table_cell_style), Paragraph("Kaggle Steel Energy CSV", table_cell_style), Paragraph("Prophet / XGBoost Reg.", table_cell_style), Paragraph("Easy", table_cell_style), Paragraph("⭐⭐⭐⭐ (Green ESG)", table_cell_style)],
        [Paragraph("5", table_cell_style), Paragraph("Raw Material Stockouts", table_cell_style), Paragraph("Supply Chain CSV Logs", table_cell_style), Paragraph("Regression / Forecast", table_cell_style), Paragraph("Easy", table_cell_style), Paragraph("⭐⭐⭐⭐ (Business)", table_cell_style)],
        [Paragraph("6", table_cell_style), Paragraph("Packaging Verifier", table_cell_style), Paragraph("Laptop Webcam (Table objects)", table_cell_style), Paragraph("Pre-trained YOLOv8 COCO", table_cell_style), Paragraph("Easy-Medium", table_cell_style), Paragraph("⭐⭐⭐⭐⭐ (Visual)", table_cell_style)],
        [Paragraph("7", table_cell_style), Paragraph("Operator Fatigue Guard", table_cell_style), Paragraph("Laptop Webcam (Live eyes)", table_cell_style), Paragraph("MediaPipe Face Mesh", table_cell_style), Paragraph("Medium", table_cell_style), Paragraph("⭐⭐⭐⭐⭐ (Dramatic)", table_cell_style)],
        [Paragraph("8", table_cell_style), Paragraph("Demand & Batch Optimizer", table_cell_style), Paragraph("Kaggle Demand CSV", table_cell_style), Paragraph("XGBoost + EOQ Algorithm", table_cell_style), Paragraph("Easy", table_cell_style), Paragraph("⭐⭐⭐⭐ (Economics)", table_cell_style)],
        [Paragraph("9", table_cell_style), Paragraph("Smart Cell Digital Twin", table_cell_style), Paragraph("Simulated Multi-station stream", table_cell_style), Paragraph("OEE Math + Anomaly AI", table_cell_style), Paragraph("Medium", table_cell_style), Paragraph("⭐⭐⭐⭐⭐ (Industry 4.0)", table_cell_style)],
        [Paragraph("10", table_cell_style), Paragraph("Scrap Rate Optimizer", table_cell_style), Paragraph("Kaggle Process Logs CSV", table_cell_style), Paragraph("Random Forest + Optimizer", table_cell_style), Paragraph("Easy-Medium", table_cell_style), Paragraph("⭐⭐⭐⭐⭐ (Cost Saving)", table_cell_style)],
    ]

    col_widths = [22, 130, 115, 115, 56, 90]
    matrix_table = Table(matrix_data, colWidths=col_widths, repeatRows=1)
    matrix_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COL),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]),
    ]))
    story.append(matrix_table)
    story.append(Spacer(1, 14))

    # ================= 48-HOUR IMPLEMENTATION ROADMAP =================
    story.append(Paragraph("Recommended 48-Hour Implementation Checklist for Laptop", h1_style))
    roadmap_html = """
    <b>Phase 1: Environment Setup (1 Hour)</b><br/>
    • Install core stack on your laptop: <code>pip install streamlit plotly pandas scikit-learn opencv-python ultralytics</code>.<br/>
    • Verify your laptop webcam or download your 1-click Kaggle dataset.<br/><br/>
    <b>Phase 2: Model Training & Validation (3-4 Hours)</b><br/>
    • For vision projects (PPE / Defect / Packaging): Use pre-trained <code>YOLOv8n.pt</code> or fine-tune on Kaggle images using Google Colab's free GPU, then export the lightweight model file back to your laptop.<br/>
    • For tabular projects (Predictive Maintenance / Bottleneck / Energy): Train <code>XGBoost</code> or <code>RandomForest</code> directly on your laptop CPU in under 60 seconds.<br/><br/>
    <b>Phase 3: Live Interactive Dashboard (4-5 Hours)</b><br/>
    • Build the interface in <code>Streamlit</code>: Add live camera feed (via <code>st.camera_input</code> or OpenCV stream) or streaming charts (via <code>plotly</code>).<br/>
    • Implement the <b>Judge Control Panel</b>: Add interactive fault injection buttons, threshold sliders, and visual green/red status badges.<br/><br/>
    <b>Phase 4: Practice the 2-Minute Pitch (1 Hour)</b><br/>
    • Rehearse the presentation: Open with the dollar/safety cost of the problem, trigger the live interactive demo, show the AI reaction, and explain the real-world factory action.
    """
    roadmap_table = Table([[Paragraph(roadmap_html, body_style)]], colWidths=[letter[0] - 108])
    roadmap_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F0FFF4")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#9AE6B4")),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(roadmap_table)

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF: {filename}")

if __name__ == '__main__':
    build_pdf()
