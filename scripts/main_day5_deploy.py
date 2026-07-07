# main_day5_deploy.py
import logging
from pathlib import Path
import gradio as gr
import pandas as pd
from ultralytics import YOLO
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("Day5Deployment")

class AutonomousPOSRegister:
    """Compiles transaction balances and displays tracking predictions within an interactive dashboard."""
    def __init__(self, model_weights: Path, catalog_prices_csv: Path):
        self.model = YOLO(str(model_weights))
        if catalog_prices_csv.exists():
            df = pd.read_csv(catalog_prices_csv)
            self.price_sheet = dict(zip(df['product_name'], df['price_usd']))
            logger.info("Price reference catalog loaded successfully.")
        else:
            logger.warning("Catalog pricing reference absent. Defaulting to base rate ($1.75).")
            self.price_sheet = {}

    def compile_transaction_bill(self, frame_rgb: np.ndarray) -> tuple:
        if frame_rgb is None:
            return None, "### ⚠️ System Warning\nNo input feed recognized."

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        results = self.model.predict(source=frame_bgr, conf=0.25, verbose=False)
        result = results[0]
        
        detected_names = [self.model.names.get(int(box.cls[0]), "Unknown") for box in result.boxes]
        confidences = [float(box.conf[0]) for box in result.boxes]
        annotated_frame = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)

        if not detected_names:
            return annotated_frame, "## 🧾 Automated Checkout Statement\n\n### 🛒 Basket Status: Empty\nPlace items on the tray."

        df = pd.DataFrame({"Item": detected_names, "Confidence": confidences})
        summary = df.groupby("Item").agg(Qty=("Item", "size"), AvgConf=("Confidence", "mean")).to_dict(orient="index")

        receipt_md = "## 🧾 Automated Checkout Statement\n\n"
        receipt_md += "| Product Class | Quantity | Confidence | Unit Price | Subtotal |\n| :--- | :---: | :---: | :---: | :---: |\n"
        
        total_balance = 0.00
        for name, metrics in summary.items():
            unit_price = self.price_sheet.get(name, 1.75)
            subtotal = metrics["Qty"] * unit_price
            total_balance += subtotal
            receipt_md += f"| **{name}** | {metrics['Qty']} | {metrics['AvgConf']:.1%} | ${unit_price:.2f} | ${subtotal:.2f} |\n"

        receipt_md += f"\n### 💵 Total Balance Due: ${total_balance:.2f}\n"
        return annotated_frame, receipt_md

    def launch_interface(self):
        with gr.Blocks(theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 🛒 SmartCart AI Assistant Dashboard")
            with gr.Row():
                with gr.Column():
                    input_img = gr.Image(type="numpy", label="Basket Intake Camera Feed")
                    checkout_btn = gr.Button("Calculate Checkout Transaction", variant="primary")
                with gr.Column():
                    output_img = gr.Image(label="YOLO11 Target Tracking Verification")
                    output_receipt = gr.Markdown(label="Compiled Digital Statement Balance")
            
            checkout_btn.click(
                fn=self.compile_transaction_bill, 
                inputs=input_img, 
                outputs=[output_img, output_receipt]
            )
        demo.launch()

if __name__ == "__main__":
    # app = AutonomousPOSRegister(Path("yolo11n.pt"), Path("./artifacts/catalog_prices.csv"))
    # app.launch_interface()
    pass
