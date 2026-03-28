import logging
logging.getLogger("prophet.plot").disabled = True
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import threading

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False


# ─────────────────────────────────────────────────────
# TOOLTIP
# ─────────────────────────────────────────────────────
class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text   = text
        self.tip    = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, event=None):
        x = self.widget.winfo_rootx() + 24
        y = self.widget.winfo_rooty() + 20
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        frame = tk.Frame(self.tip, background="#FFFDE7", relief="solid", borderwidth=1)
        frame.pack()
        tk.Label(frame, text=self.text, background="#FFFDE7", foreground="#333333",
                 font=("Arial", 9), justify="left", wraplength=340, padx=8, pady=6).pack()

    def hide(self, event=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


def tip_btn(parent, text, **grid_kwargs):
    lbl = tk.Label(parent, text=" ? ", font=("Arial", 8, "bold"),
                   fg="white", bg="#5B9BD5", cursor="question_arrow",
                   relief="flat", padx=2)
    lbl.grid(**grid_kwargs)
    Tooltip(lbl, text)
    return lbl


# ─────────────────────────────────────────────────────
# TOOLTIP TEXTS
# ─────────────────────────────────────────────────────
TIP = {
    "freq": (
        "Data Frequency\n\n"
        "Select the frequency you want to forecast at:\n\n"
        "  • Daily     → one period per day\n"
        "  • Weekly    → one period per week\n"
        "  • Monthly   → one period per month\n"
        "  • Quarterly → one period per quarter\n"
        "  • Yearly    → one period per year\n\n"
        "This controls how the data is grouped\n"
        "and what S is suggested for SARIMA."
    ),
    "agg": (
        "Aggregation Method\n\n"
        "How to combine multiple raw rows\n"
        "into one period:\n\n"
        "  • None → use data exactly as imported\n"
        "  • Sum  → total of all values in period\n"
        "           (best for sales/units/demand)\n"
        "  • Mean → average of values in period\n"
        "           (best for prices/rates)\n"
        "  • Max  → highest value in period\n"
        "  • Min  → lowest value in period\n\n"
        "Example: daily sales + Monthly + Sum\n"
        "= total monthly sales."
    ),
    "S": (
        "S – Seasonal Period\n\n"
        "  • Monthly data   → S = 12\n"
        "  • Quarterly data → S = 4\n"
        "  • Weekly data    → S = 52\n"
        "  • Daily data     → S = 7 (weekly cycle)\n\n"
        "Look at the plot: does the pattern repeat\n"
        "every 12 months? Use 12."
    ),
    "D": (
        "D – Seasonal Differencing\n\n"
        "  • D=1: most common, removes seasonal waves\n"
        "  • D=0: no seasonal pattern in data\n"
        "  • D=2: rarely needed\n\n"
        "Click 'Plot seasonal diff' and check if\n"
        "the result looks flat with no waves."
    ),
    "d": (
        "d – Non-Seasonal Differencing\n\n"
        "  • d=1: most common, removes trend\n"
        "  • d=0: data already has no trend\n"
        "  • d=2: only if d=1 still shows trend\n\n"
        "Click 'Plot differenced' — result should\n"
        "hover around zero with no drift."
    ),
    "p": (
        "p – AR Order  (from PACF)\n\n"
        "  • Plot PACF\n"
        "  • Count bars sticking OUTSIDE the bands\n"
        "    starting from lag 1\n"
        "  • Last outside lag = p\n\n"
        "Tip: p = 0, 1, or 2 covers most cases."
    ),
    "P": (
        "P – Seasonal AR Order  (from PACF)\n\n"
        "  • Look at PACF bars at lag S, 2S, 3S\n"
        "  • Lag S outside but 2S inside → P = 1\n"
        "  • Both S and 2S outside → P = 2\n\n"
        "Tip: P = 0 or 1 almost always enough."
    ),
    "q": (
        "q – MA Order  (from ACF)\n\n"
        "  • Plot ACF\n"
        "  • Count bars sticking OUTSIDE the bands\n"
        "    starting from lag 1\n"
        "  • Last outside lag = q\n\n"
        "Tip: q = 0, 1, or 2 covers most cases."
    ),
    "Q": (
        "Q – Seasonal MA Order  (from ACF)\n\n"
        "  • Look at ACF bars at lag S, 2S, 3S\n"
        "  • Lag S outside but 2S inside → Q = 1\n"
        "  • Both S and 2S outside → Q = 2\n\n"
        "Tip: Q = 0 or 1 almost always enough."
    ),
    "cps": (
        "Trend Flexibility\n\n"
        "  • Low  (0.01–0.05): smooth stable trend\n"
        "  • High (0.1–0.5):  flexible trend\n\n"
        "Default: 0.05\n"
        "Increase if forecast looks too rigid.\n"
        "Decrease if forecast looks too erratic."
    ),
    "sps": (
        "Seasonality Strength\n\n"
        "  • Low  (0.01–1.0): smooth seasonality\n"
        "  • High (5.0–20.0): strong seasonality\n\n"
        "Default: 10.0\n"
        "Increase if seasonal peaks look too small.\n"
        "Decrease if too exaggerated."
    ),
}

FREQ_CONFIG = {
    "Daily":     ("D",  "D",  7),
    "Weekly":    ("W",  "W",  52),
    "Monthly":   ("MS", "MS", 12),
    "Quarterly": ("QS", "QS", 4),
    "Yearly":    ("YS", "Y",  1),
}


class ForecastApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Demand Forecasting App – ARIMA/SARIMA/Prophet")
        self.root.geometry("780x960")
        self.df      = None
        self.raw_ts  = None   # original imported series
        self.ts      = None   # grouped/active series used for modelling
        self.forecast_result = None
        self.build_ui()

    # ─────────────────────────────────────────────
    # ROBUST DATE PARSER
    # ─────────────────────────────────────────────
    def parse_dates_robust(self, series):
        formats = [
            "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y",
            "%b-%y", "%b-%Y", "%B-%Y", "%b %Y", "%B %Y",
            "%m/%Y", "%Y-%m", "%Y%m", "%d %b %Y", "%d %B %Y",
        ]
        for fmt in formats:
            try:
                return pd.to_datetime(series, format=fmt)
            except Exception:
                continue
        try:
            return pd.to_datetime(series, infer_datetime_format=True, dayfirst=True)
        except Exception:
            raise ValueError(
                f"Cannot parse dates.\nExample: '{series.iloc[0]}'\n"
                "Supported: dd/mm/yyyy, Jan-17, 2017-01, Jan 2017, etc."
            )

    # ─────────────────────────────────────────────
    # UI BUILD
    # ─────────────────────────────────────────────
    def build_ui(self):
        pad = {"padx": 4, "pady": 3}

        canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        self.scroll_frame = ttk.Frame(canvas, padding=12)
        self.scroll_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        main = self.scroll_frame

        # ══════════════════════════════════════════
        # STAGE 1 – IMPORT
        # ══════════════════════════════════════════
        ttk.Label(main, text="── STAGE 1 : IMPORT DATA ──",
                  font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 2))

        imp = ttk.Frame(main); imp.pack(fill="x")
        ttk.Button(imp, text="📁 Import CSV", command=self.import_csv).pack(side="left")
        ttk.Button(imp, text="🔄 Refresh App", command=self.refresh_app).pack(side="left", padx=(8, 0))
        self.import_label = ttk.Label(imp, text="No file loaded.", foreground="gray")
        self.import_label.pack(side="left", padx=10)

        col = ttk.Frame(main); col.pack(fill="x", pady=4)
        ttk.Label(col, text="Date column:").grid(row=0, column=0, sticky="w", **pad)
        self.date_col_var = tk.StringVar()
        self.date_col_cb = ttk.Combobox(col, textvariable=self.date_col_var, state="disabled", width=16)
        self.date_col_cb.grid(row=0, column=1, sticky="w", **pad)
        ttk.Label(col, text="Forecast column:").grid(row=0, column=2, sticky="w", **pad)
        self.forecast_col_var = tk.StringVar()
        self.forecast_col_cb = ttk.Combobox(col, textvariable=self.forecast_col_var, state="disabled", width=16)
        self.forecast_col_cb.grid(row=0, column=3, sticky="w", **pad)
        self.confirm_col_btn = ttk.Button(col, text="✅ Confirm columns",
                                          command=self.confirm_columns, state="disabled")
        self.confirm_col_btn.grid(row=0, column=4, **pad)

        # raw data status
        self.raw_status_label = ttk.Label(main, text="", foreground="darkgreen", font=("Arial", 9))
        self.raw_status_label.pack(anchor="w", padx=4)

        ttk.Button(main, text="📈 Plot original data", command=self.plot_original).pack(anchor="w", pady=(4, 0))

        # ══════════════════════════════════════════
        # STAGE 2 – GROUPING
        # ══════════════════════════════════════════
        ttk.Label(main, text="── STAGE 2 : GROUPING ──",
                  font=("Arial", 10, "bold")).pack(anchor="w", pady=(10, 2))

        grp = ttk.LabelFrame(main, text="Frequency & Aggregation", padding=6)
        grp.pack(fill="x", pady=2)
        gr = ttk.Frame(grp); gr.pack(fill="x")

        ttk.Label(gr, text="Frequency:").grid(row=0, column=0, sticky="w", **pad)
        self.freq_var = tk.StringVar(value="Monthly")
        freq_cb = ttk.Combobox(gr, textvariable=self.freq_var,
                               values=list(FREQ_CONFIG.keys()), state="readonly", width=10)
        freq_cb.grid(row=0, column=1, **pad)
        freq_cb.bind("<<ComboboxSelected>>", self.on_freq_change)
        tip_btn(gr, TIP["freq"], row=0, column=2, **pad)

        ttk.Label(gr, text="Aggregate by:").grid(row=0, column=3, sticky="w", **pad)
        self.agg_var = tk.StringVar(value="None")
        agg_cb = ttk.Combobox(gr, textvariable=self.agg_var,
                              values=["None", "Sum", "Mean", "Max", "Min"],
                              state="readonly", width=8)
        agg_cb.grid(row=0, column=4, **pad)
        tip_btn(gr, TIP["agg"], row=0, column=5, **pad)

        # Confirm grouping + show grouped buttons
        gr2 = ttk.Frame(grp); gr2.pack(fill="x", pady=(4, 0))
        self.confirm_grp_btn = ttk.Button(gr2, text="✅ Confirm Grouping",
                                          command=self.confirm_grouping, state="disabled")
        self.confirm_grp_btn.pack(side="left")
        self.show_grp_btn = ttk.Button(gr2, text="📊 Show grouped vs original",
                                       command=self.plot_grouped_vs_original, state="disabled")
        self.show_grp_btn.pack(side="left", padx=(8, 0))

        # Grouping status
        self.grp_status_label = ttk.Label(grp, text="No grouping confirmed yet.",
                                          foreground="gray", font=("Arial", 9))
        self.grp_status_label.pack(anchor="w", pady=(4, 0))

        # ══════════════════════════════════════════
        # STAGE 3 – MODEL
        # ══════════════════════════════════════════
        ttk.Label(main, text="── STAGE 3 : MODEL ──",
                  font=("Arial", 10, "bold")).pack(anchor="w", pady=(10, 2))

        mod = ttk.Frame(main); mod.pack(fill="x")
        ttk.Label(mod, text="Model:").pack(side="left")
        self.model_var = tk.StringVar(value="SARIMA")
        model_cb = ttk.Combobox(mod, textvariable=self.model_var,
                                values=["ARIMA", "SARIMA", "Prophet"], state="readonly", width=10)
        model_cb.pack(side="left", padx=6)
        model_cb.bind("<<ComboboxSelected>>", self.toggle_model)

        # Active data badge
        self.active_data_label = ttk.Label(mod, text="", foreground="darkblue",
                                           font=("Arial", 9, "italic"))
        self.active_data_label.pack(side="left", padx=12)

        # ── STEP 1: SEASONAL DIFF ─────────────────
        self.seasonal_frame = ttk.LabelFrame(main,
            text="STEP 1 – Seasonal Differencing  (SARIMA only)", padding=6)
        s1 = ttk.Frame(self.seasonal_frame); s1.pack(fill="x")
        ttk.Label(s1, text="S:").grid(row=0, column=0, sticky="w", **pad)
        self.S_var = tk.StringVar(value="12")
        self.S_entry = ttk.Entry(s1, textvariable=self.S_var, width=5)
        self.S_entry.grid(row=0, column=1, **pad)
        tip_btn(s1, TIP["S"], row=0, column=2, **pad)
        ttk.Label(s1, text="D:").grid(row=0, column=3, sticky="w", **pad)
        self.D_var = tk.StringVar(value="1")
        self.D_entry = ttk.Entry(s1, textvariable=self.D_var, width=5)
        self.D_entry.grid(row=0, column=4, **pad)
        tip_btn(s1, TIP["D"], row=0, column=5, **pad)
        self.plot_seas_btn = ttk.Button(s1, text="Plot seasonal diff",
                                        command=self.plot_seasonal_diff)
        self.plot_seas_btn.grid(row=0, column=6, **pad)

        # ── STEP 2: NON-SEASONAL DIFF ─────────────
        self.step2_frame = ttk.LabelFrame(main,
            text="STEP 2 – Non-Seasonal Differencing", padding=6)
        s2 = ttk.Frame(self.step2_frame); s2.pack(fill="x")
        ttk.Label(s2, text="d:").grid(row=0, column=0, sticky="w", **pad)
        self.d_var = tk.StringVar(value="1")
        ttk.Entry(s2, textvariable=self.d_var, width=5).grid(row=0, column=1, **pad)
        tip_btn(s2, TIP["d"], row=0, column=2, **pad)
        ttk.Button(s2, text="Plot differenced",
                   command=self.plot_nonseasonal_diff).grid(row=0, column=3, **pad)

        # ── STEP 3: AR TERMS ──────────────────────
        self.step3_frame = ttk.LabelFrame(main,
            text="STEP 3 – AR Terms: use PACF → p  (and P for SARIMA)", padding=6)
        s3 = ttk.Frame(self.step3_frame); s3.pack(fill="x")
        ttk.Button(s3, text="Plot PACF", command=self.plot_pacf_).grid(row=0, column=0, **pad)
        ttk.Label(s3, text="p:").grid(row=0, column=1, sticky="w", **pad)
        self.p_var = tk.StringVar(value="1")
        ttk.Entry(s3, textvariable=self.p_var, width=5).grid(row=0, column=2, **pad)
        tip_btn(s3, TIP["p"], row=0, column=3, **pad)
        ttk.Label(s3, text="P:").grid(row=0, column=4, sticky="w", **pad)
        self.P_var = tk.StringVar(value="1")
        self.P_entry = ttk.Entry(s3, textvariable=self.P_var, width=5)
        self.P_entry.grid(row=0, column=5, **pad)
        tip_btn(s3, TIP["P"], row=0, column=6, **pad)

        # ── STEP 4: MA TERMS ──────────────────────
        self.step4_frame = ttk.LabelFrame(main,
            text="STEP 4 – MA Terms: use ACF → q  (and Q for SARIMA)", padding=6)
        s4 = ttk.Frame(self.step4_frame); s4.pack(fill="x")
        ttk.Button(s4, text="Plot ACF", command=self.plot_acf_).grid(row=0, column=0, **pad)
        ttk.Label(s4, text="q:").grid(row=0, column=1, sticky="w", **pad)
        self.q_var = tk.StringVar(value="1")
        ttk.Entry(s4, textvariable=self.q_var, width=5).grid(row=0, column=2, **pad)
        tip_btn(s4, TIP["q"], row=0, column=3, **pad)
        ttk.Label(s4, text="Q:").grid(row=0, column=4, sticky="w", **pad)
        self.Q_var = tk.StringVar(value="1")
        self.Q_entry = ttk.Entry(s4, textvariable=self.Q_var, width=5)
        self.Q_entry.grid(row=0, column=5, **pad)
        tip_btn(s4, TIP["Q"], row=0, column=6, **pad)

        # ── AUTO SELECT ───────────────────────────
        self.auto_frame = ttk.LabelFrame(main,
            text="⚡ Auto Select Parameters (ARIMA/SARIMA)", padding=6)
        af = ttk.Frame(self.auto_frame); af.pack(fill="x")
        self.auto_btn = ttk.Button(af,
            text="🔍 Auto select p,d,q  (and P,D,Q for SARIMA)",
            command=self.run_auto_arima)
        self.auto_btn.pack(side="left")
        self.auto_status_label = ttk.Label(af, text="", foreground="green")
        self.auto_status_label.pack(side="left", padx=10)
        self.auto_result_label = ttk.Label(self.auto_frame, text="Auto result: –",
                                           font=("Courier", 10), foreground="darkblue")
        self.auto_result_label.pack(anchor="w", pady=(4, 0))

        # ── PROPHET SETTINGS ──────────────────────
        self.prophet_frame = ttk.LabelFrame(main,
            text="STEP 1–4 – Prophet Settings", padding=6)
        pf = ttk.Frame(self.prophet_frame); pf.pack(fill="x")
        ttk.Label(pf, text="Yearly seasonality:").grid(row=0, column=0, sticky="w", **pad)
        self.prophet_yearly_var = tk.StringVar(value="auto")
        ttk.Combobox(pf, textvariable=self.prophet_yearly_var,
                     values=["auto", "True", "False"],
                     state="readonly", width=7).grid(row=0, column=1, **pad)
        ttk.Label(pf, text="Weekly seasonality:").grid(row=0, column=2, sticky="w", **pad)
        self.prophet_weekly_var = tk.StringVar(value="auto")
        ttk.Combobox(pf, textvariable=self.prophet_weekly_var,
                     values=["auto", "True", "False"],
                     state="readonly", width=7).grid(row=0, column=3, **pad)
        pf2 = ttk.Frame(self.prophet_frame); pf2.pack(fill="x")
        ttk.Label(pf2, text="Trend flexibility:").grid(row=0, column=0, sticky="w", **pad)
        self.prophet_cps_var = tk.StringVar(value="0.05")
        ttk.Entry(pf2, textvariable=self.prophet_cps_var, width=7).grid(row=0, column=1, **pad)
        tip_btn(pf2, TIP["cps"], row=0, column=2, **pad)
        pf3 = ttk.Frame(self.prophet_frame); pf3.pack(fill="x")
        ttk.Label(pf3, text="Seasonality strength:").grid(row=0, column=0, sticky="w", **pad)
        self.prophet_sps_var = tk.StringVar(value="10.0")
        ttk.Entry(pf3, textvariable=self.prophet_sps_var, width=7).grid(row=0, column=1, **pad)
        tip_btn(pf3, TIP["sps"], row=0, column=2, **pad)

        # ── STEP 5: VALIDATION ────────────────────
        self.step5_frame = ttk.LabelFrame(main,
            text="STEP 5 – Model Validation (optional)", padding=6)
        s5 = ttk.Frame(self.step5_frame); s5.pack(fill="x")
        self.test_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(s5, text="Test model", variable=self.test_var).pack(side="left")
        ttk.Label(s5, text="Last test periods:").pack(side="left", padx=(10, 4))
        self.test_months_var = tk.StringVar(value="6")
        ttk.Entry(s5, textvariable=self.test_months_var, width=5).pack(side="left")
        ttk.Button(s5, text="Run test", command=self.run_test).pack(side="left", padx=8)
        self.metrics_label = ttk.Label(self.step5_frame,
            text="MAPE: --   RMSE: --   MAE: --", foreground="blue")
        self.metrics_label.pack(anchor="w", pady=2)

        # ── STEP 6: FORECAST ──────────────────────
        self.step6_frame = ttk.LabelFrame(main, text="STEP 6 – Forecast", padding=6)
        s6 = ttk.Frame(self.step6_frame); s6.pack(fill="x")
        ttk.Label(s6, text="Periods ahead:").pack(side="left")
        self.forecast_periods_var = tk.StringVar(value="12")
        ttk.Entry(s6, textvariable=self.forecast_periods_var, width=5).pack(side="left", padx=4)
        self.periods_unit_label = ttk.Label(s6, text="(months)", foreground="gray")
        self.periods_unit_label.pack(side="left")
        ttk.Button(s6, text="▶ Run forecast", command=self.run_forecast).pack(side="left", padx=8)
        ttk.Button(s6, text="💾 Save CSV", command=self.save_csv).pack(side="left")

        self.all_dynamic_frames = [
            self.seasonal_frame, self.step2_frame, self.step3_frame,
            self.step4_frame, self.auto_frame, self.prophet_frame,
            self.step5_frame, self.step6_frame,
        ]
        self.toggle_model()

    # ─────────────────────────────────────────────
    # FREQ CHANGE
    # ─────────────────────────────────────────────
    def on_freq_change(self, event=None):
        freq = self.freq_var.get()
        _, _, s = FREQ_CONFIG[freq]
        self.S_var.set(str(s))
        unit_map = {"Daily": "(days)", "Weekly": "(weeks)",
                    "Monthly": "(months)", "Quarterly": "(quarters)", "Yearly": "(years)"}
        self.periods_unit_label.config(text=unit_map.get(freq, ""))

    def get_prophet_freq(self):
        return FREQ_CONFIG[self.freq_var.get()][1]

    def get_resample_rule(self):
        return FREQ_CONFIG[self.freq_var.get()][0]

    # ─────────────────────────────────────────────
    # TOGGLE MODEL
    # ─────────────────────────────────────────────
    def toggle_model(self, event=None):
        model      = self.model_var.get()
        is_sarima  = model == "SARIMA"
        is_prophet = model == "Prophet"

        for frame in self.all_dynamic_frames:
            frame.pack_forget()

        if is_sarima:
            self.seasonal_frame.pack(fill="x", pady=4)
        if not is_prophet:
            self.step2_frame.pack(fill="x", pady=4)
            self.step3_frame.pack(fill="x", pady=4)
            self.step4_frame.pack(fill="x", pady=4)
            self.auto_frame.pack(fill="x", pady=4)
        if is_prophet:
            self.prophet_frame.pack(fill="x", pady=4)

        self.step5_frame.pack(fill="x", pady=4)
        self.step6_frame.pack(fill="x", pady=4)

        pq_state = "normal" if is_sarima else "disabled"
        self.P_entry.configure(state=pq_state)
        self.Q_entry.configure(state=pq_state)

    # ─────────────────────────────────────────────
    # AUTO ARIMA
    # ─────────────────────────────────────────────
    def run_auto_arima(self):
        if not self.check_ts(): return
        if not PMDARIMA_AVAILABLE:
            messagebox.showerror("Missing package",
                "pmdarima not installed.\nRun:\n\n  pip install pmdarima"); return
        self.auto_btn.configure(state="disabled")
        self.auto_status_label.config(text="⏳ Searching...", foreground="orange")
        self.auto_result_label.config(text="Auto result: searching...")
        threading.Thread(target=self._auto_arima_thread, daemon=True).start()

    def _auto_arima_thread(self):
        try:
            is_sarima = self.model_var.get() == "SARIMA"
            S = int(self.S_var.get()) if is_sarima else 1
            model = auto_arima(self.ts, seasonal=is_sarima, m=S,
                               stepwise=True, suppress_warnings=True,
                               error_action="ignore", information_criterion="aic")
            p, d, q = model.order
            P, D, Q, _ = model.seasonal_order if is_sarima else (0, 0, 0, 0)
            self.root.after(0, lambda: self._auto_arima_done(p, d, q, P, D, Q, is_sarima))
        except Exception as e:
            self.root.after(0, lambda: self._auto_arima_error(str(e)))

    def _auto_arima_done(self, p, d, q, P, D, Q, is_sarima):
        self.p_var.set(p); self.d_var.set(d); self.q_var.set(q)
        if is_sarima:
            self.P_var.set(P); self.D_var.set(D); self.Q_var.set(Q)
        txt = (f"Auto result:  SARIMA ({p},{d},{q})({P},{D},{Q},{self.S_var.get()})  ← filled"
               if is_sarima else f"Auto result:  ARIMA ({p},{d},{q})  ← filled")
        self.auto_result_label.config(text=txt)
        self.auto_status_label.config(text="✅ Done.", foreground="green")
        self.auto_btn.configure(state="normal")

    def _auto_arima_error(self, msg):
        self.auto_status_label.config(text="❌ Failed.", foreground="red")
        self.auto_result_label.config(text="Auto result: error")
        self.auto_btn.configure(state="normal")
        messagebox.showerror("Auto ARIMA Error", msg)

    # ─────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────
    def clean_numeric(self, series):
        if pd.api.types.is_numeric_dtype(series):
            return series.astype(float)
        return pd.to_numeric(
            series.astype(str).str.replace(r'[\$,\s]', '', regex=True), errors='coerce')

    def get_final_series(self):
        s = self.ts.copy().dropna()
        if self.model_var.get() == "SARIMA":
            S, D = int(self.S_var.get()), int(self.D_var.get())
            for _ in range(D): s = s.diff(S).dropna()
        for _ in range(int(self.d_var.get())): s = s.diff(1).dropna()
        return s

    def get_orders(self):
        p, d, q = int(self.p_var.get()), int(self.d_var.get()), int(self.q_var.get())
        if self.model_var.get() == "SARIMA":
            return (p, d, q), (int(self.P_var.get()), int(self.D_var.get()),
                               int(self.Q_var.get()), int(self.S_var.get()))
        return (p, d, q), (0, 0, 0, 0)

    def check_ts(self):
        if self.ts is None:
            messagebox.showwarning("Warning",
                "No active data.\nComplete Stage 1 (confirm columns)\n"
                "and Stage 2 (confirm grouping) first.")
            return False
        return True

    def get_prophet_params(self):
        def pb(v): return True if v == "True" else (False if v == "False" else "auto")
        return {
            "yearly_seasonality": pb(self.prophet_yearly_var.get()),
            "weekly_seasonality": pb(self.prophet_weekly_var.get()),
            "changepoint_prior_scale": float(self.prophet_cps_var.get()),
            "seasonality_prior_scale": float(self.prophet_sps_var.get()),
        }

    def ts_to_prophet_df(self, series):
        df = series.reset_index(); df.columns = ["ds", "y"]
        df["ds"] = pd.to_datetime(df["ds"]); return df

    # ─────────────────────────────────────────────
    # REFRESH
    # ─────────────────────────────────────────────
    def refresh_app(self):
        if not messagebox.askyesno("Refresh", "Reset everything and start fresh?"): return
        self.df = None; self.raw_ts = None; self.ts = None
        self.forecast_result = None
        self.import_label.config(text="No file loaded.", foreground="gray")
        self.raw_status_label.config(text="")
        self.grp_status_label.config(text="No grouping confirmed yet.", foreground="gray")
        self.active_data_label.config(text="")
        for cb in (self.date_col_cb, self.forecast_col_cb):
            cb.set(""); cb["values"] = []; cb["state"] = "disabled"
        self.confirm_col_btn.configure(state="disabled")
        self.confirm_grp_btn.configure(state="disabled")
        self.show_grp_btn.configure(state="disabled")
        self.model_var.set("SARIMA")
        self.freq_var.set("Monthly"); self.agg_var.set("None")
        self.periods_unit_label.config(text="(months)")
        self.S_var.set("12"); self.D_var.set("1"); self.d_var.set("1")
        self.p_var.set("1");  self.P_var.set("1")
        self.q_var.set("1");  self.Q_var.set("1")
        self.prophet_yearly_var.set("auto"); self.prophet_weekly_var.set("auto")
        self.prophet_cps_var.set("0.05");   self.prophet_sps_var.set("10.0")
        self.test_var.set(False); self.test_months_var.set("6")
        self.forecast_periods_var.set("12")
        self.metrics_label.config(text="MAPE: --   RMSE: --   MAE: --")
        self.auto_result_label.config(text="Auto result: –")
        self.auto_status_label.config(text="")
        plt.close("all"); self.toggle_model()
        messagebox.showinfo("Refreshed", "App reset. Ready for a new file.")

    # ─────────────────────────────────────────────
    # STAGE 1 — IMPORT & CONFIRM COLUMNS
    # ─────────────────────────────────────────────
    def import_csv(self):
        fp = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not fp: return
        try:
            self.df = pd.read_csv(fp)
            cols = list(self.df.columns)
            for cb in (self.date_col_cb, self.forecast_col_cb):
                cb["values"] = cols; cb["state"] = "readonly"
            self.confirm_col_btn.configure(state="normal")
            self.import_label.config(
                text=f"Loaded: {len(self.df)} rows · {len(cols)} columns", foreground="black")
        except Exception as e:
            messagebox.showerror("Import Error", str(e))

    def confirm_columns(self):
        dc, fc = self.date_col_var.get(), self.forecast_col_var.get()
        if not dc or not fc:
            messagebox.showwarning("Warning", "Select both columns first."); return
        try:
            dates  = self.parse_dates_robust(self.df[dc].astype(str).str.strip())
            values = self.clean_numeric(self.df[fc])
            if values.isna().all():
                messagebox.showerror("Error", f"No valid numbers in '{fc}'."); return
            tmp = pd.DataFrame({"date": dates, "value": values}).dropna()
            tmp.sort_values("date", inplace=True)
            self.raw_ts = tmp.set_index("date")["value"]

            # Default: ts = raw until grouping is confirmed
            self.ts = self.raw_ts.copy()

            self.raw_status_label.config(
                text=f"  ✅ Raw data loaded: {len(self.raw_ts)} rows  |  "
                     f"{self.raw_ts.index.min().date()} → {self.raw_ts.index.max().date()}",
                foreground="darkgreen")
            self.confirm_grp_btn.configure(state="normal")
            self.active_data_label.config(
                text=f"Using: raw data ({len(self.ts)} pts)")
        except Exception as e:
            messagebox.showerror("Column Error", str(e))

    # ─────────────────────────────────────────────
    # STAGE 2 — CONFIRM GROUPING
    # ─────────────────────────────────────────────
    def confirm_grouping(self):
        if self.raw_ts is None:
            messagebox.showwarning("Warning", "Confirm columns first."); return
        try:
            agg  = self.agg_var.get()
            freq = self.freq_var.get()
            rule = self.get_resample_rule()

            if agg == "None":
                self.ts = self.raw_ts.copy()
                status = (f"  ✅ No aggregation — using raw data as-is  "
                          f"({len(self.ts)} rows)")
            else:
                agg_map = {"Sum": "sum", "Mean": "mean", "Max": "max", "Min": "min"}
                self.ts = getattr(self.raw_ts.resample(rule), agg_map[agg])().dropna()
                status = (f"  ✅ Grouped: {freq} · {agg}  |  "
                          f"{len(self.raw_ts)} raw rows → {len(self.ts)} {freq} periods  |  "
                          f"{self.ts.index.min().date()} → {self.ts.index.max().date()}")

            self.grp_status_label.config(text=status, foreground="darkblue")
            self.show_grp_btn.configure(state="normal")
            self.active_data_label.config(
                text=f"Using: {freq} {agg} ({len(self.ts)} pts)")

            # Auto-update S
            _, _, suggested_S = FREQ_CONFIG[freq]
            self.S_var.set(str(suggested_S))
            unit_map = {"Daily": "(days)", "Weekly": "(weeks)",
                        "Monthly": "(months)", "Quarterly": "(quarters)", "Yearly": "(years)"}
            self.periods_unit_label.config(text=unit_map.get(freq, ""))

        except Exception as e:
            messagebox.showerror("Grouping Error", str(e))

    # ─────────────────────────────────────────────
    # PLOT GROUPED VS ORIGINAL
    # ─────────────────────────────────────────────
    def plot_grouped_vs_original(self):
        if self.raw_ts is None or self.ts is None:
            messagebox.showwarning("Warning", "Confirm columns and grouping first."); return
        try:
            agg  = self.agg_var.get()
            freq = self.freq_var.get()

            fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=False)
            fig.suptitle("Original Data  vs  Grouped Data", fontsize=13, fontweight="bold")

            # Top: original raw
            axes[0].plot(self.raw_ts.index, self.raw_ts.values,
                         color="steelblue", linewidth=1.2, alpha=0.85)
            axes[0].set_title(f"Original Data  ({len(self.raw_ts)} rows)", fontsize=11)
            axes[0].set_ylabel("Value"); axes[0].grid(True, alpha=0.3)

            # Bottom: grouped
            if agg == "None":
                color  = "steelblue"
                label  = "No grouping (same as original)"
            else:
                color  = "darkorange"
                label  = f"{freq} · {agg}  ({len(self.ts)} periods)"

            axes[1].bar(self.ts.index, self.ts.values,
                        color=color, alpha=0.75, width=20)
            axes[1].plot(self.ts.index, self.ts.values,
                         color=color, linewidth=1.5, marker="o", markersize=3)
            axes[1].set_title(f"Grouped Data  –  {label}", fontsize=11)
            axes[1].set_ylabel("Value"); axes[1].grid(True, alpha=0.3)

            plt.tight_layout(); plt.show()
        except Exception as e:
            messagebox.showerror("Plot Error", str(e))

    # ─────────────────────────────────────────────
    # PLOT ORIGINAL (raw)
    # ─────────────────────────────────────────────
    def plot_original(self):
        if self.raw_ts is None:
            messagebox.showwarning("Warning", "Import and confirm columns first."); return
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(self.raw_ts.index, self.raw_ts.values, color="steelblue", linewidth=1.8)
        ax.set_title(f"Original Raw Data  ({len(self.raw_ts)} rows)", fontsize=13)
        ax.set_xlabel("Date"); ax.set_ylabel("Value"); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.show()

    # ─────────────────────────────────────────────
    # DIFFERENCING PLOTS
    # ─────────────────────────────────────────────
    def plot_seasonal_diff(self):
        if not self.check_ts(): return
        try:
            S, D = int(self.S_var.get()), int(self.D_var.get())
            s = self.ts.copy().dropna()
            for _ in range(D): s = s.diff(S).dropna()
            fig, axes = plt.subplots(2, 1, figsize=(12, 7))
            axes[0].plot(self.ts.index, self.ts.values, color="steelblue", linewidth=1.5)
            axes[0].set_title("Active Series (grouped)", fontsize=12); axes[0].grid(True, alpha=0.3)
            axes[1].plot(s.index, s.values, color="darkorange", linewidth=1.5)
            axes[1].set_title(f"After Seasonal Differencing  (D={D}, S={S})", fontsize=12)
            axes[1].axhline(0, linestyle="--", color="gray", alpha=0.6)
            axes[1].grid(True, alpha=0.3)
            plt.tight_layout(); plt.show()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def plot_nonseasonal_diff(self):
        if not self.check_ts(): return
        try:
            d = int(self.d_var.get())
            start = self.ts.copy().dropna()
            if self.model_var.get() == "SARIMA":
                S, D = int(self.S_var.get()), int(self.D_var.get())
                for _ in range(D): start = start.diff(S).dropna()
            final = start.copy()
            for _ in range(d): final = final.diff(1).dropna()
            fig, axes = plt.subplots(2, 1, figsize=(12, 7))
            axes[0].plot(start.index, start.values, color="darkorange", linewidth=1.5)
            axes[0].set_title("Before Non-Seasonal Differencing", fontsize=12)
            axes[0].grid(True, alpha=0.3)
            axes[1].plot(final.index, final.values, color="green", linewidth=1.5)
            axes[1].set_title(f"Final Stationary Series  (d={d})", fontsize=12)
            axes[1].axhline(0, linestyle="--", color="gray", alpha=0.6)
            axes[1].grid(True, alpha=0.3)
            plt.tight_layout(); plt.show()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def plot_pacf_(self):
        if not self.check_ts(): return
        try:
            s = self.get_final_series()
            fig, ax = plt.subplots(figsize=(12, 5))
            plot_pacf(s, ax=ax, lags=min(40, len(s) // 2 - 1))
            ax.set_title("PACF  →  last lag outside bands = p   (at S, 2S = P)", fontsize=12)
            ax.grid(True, alpha=0.3); plt.tight_layout(); plt.show()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def plot_acf_(self):
        if not self.check_ts(): return
        try:
            s = self.get_final_series()
            fig, ax = plt.subplots(figsize=(12, 5))
            plot_acf(s, ax=ax, lags=min(40, len(s) // 2 - 1))
            ax.set_title("ACF  →  last lag outside bands = q   (at S, 2S = Q)", fontsize=12)
            ax.grid(True, alpha=0.3); plt.tight_layout(); plt.show()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ─────────────────────────────────────────────
    # TEST & FORECAST
    # ─────────────────────────────────────────────
    def run_test(self):
        if not self.check_ts(): return
        if not self.test_var.get():
            messagebox.showinfo("Info", "Tick 'Test model' first."); return
        try:
            h = int(self.test_months_var.get())
            if h >= len(self.ts):
                messagebox.showerror("Error", "Test period longer than data."); return
            train, test = self.ts.iloc[:-h], self.ts.iloc[-h:]
            if self.model_var.get() == "Prophet":
                if not PROPHET_AVAILABLE:
                    messagebox.showerror("Error", "Run: pip install prophet"); return
                m = Prophet(**self.get_prophet_params())
                m.fit(self.ts_to_prophet_df(train))
                future  = m.make_future_dataframe(periods=h, freq=self.get_prophet_freq())
                fc_vals = m.predict(future)["yhat"].iloc[-h:].values
            else:
                order, seasonal_order = self.get_orders()
                result = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                                 enforce_stationarity=False,
                                 enforce_invertibility=False).fit(disp=False)
                fc_vals = result.forecast(steps=h).values
            mae  = np.mean(np.abs(test.values - fc_vals))
            rmse = np.sqrt(np.mean((test.values - fc_vals) ** 2))
            mape = np.mean(np.abs((test.values - fc_vals) / test.values)) * 100
            self.metrics_label.config(
                text=f"MAPE: {mape:.2f}%   RMSE: {rmse:,.2f}   MAE: {mae:,.2f}")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(train.index, train.values, label="Train", color="steelblue")
            ax.plot(test.index, test.values,   label="Actual (test)", color="green")
            ax.plot(test.index, fc_vals,        label="Forecast (test)",
                    color="red", linestyle="--")
            ax.set_title(
                f"Test Forecast vs Actual  |  MAPE: {mape:.2f}%   RMSE: {rmse:,.2f}",
                fontsize=12)
            ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout(); plt.show()
        except Exception as e:
            messagebox.showerror("Model Error", str(e))

    def run_forecast(self):
        if not self.check_ts(): return
        try:
            h    = int(self.forecast_periods_var.get())
            freq = self.freq_var.get()
            if self.model_var.get() == "Prophet":
                if not PROPHET_AVAILABLE:
                    messagebox.showerror("Error", "Run: pip install prophet"); return
                m = Prophet(**self.get_prophet_params())
                m.fit(self.ts_to_prophet_df(self.ts))
                future = m.make_future_dataframe(periods=h, freq=self.get_prophet_freq())
                fc_df  = m.predict(future)
                mean   = fc_df["yhat"].iloc[-h:]
                lower  = fc_df["yhat_lower"].iloc[-h:]
                upper  = fc_df["yhat_upper"].iloc[-h:]
                idx    = pd.to_datetime(fc_df["ds"].iloc[-h:].values)
                self.forecast_result = pd.DataFrame({
                    "date": idx, "forecast": mean.values,
                    "lower_95": lower.values, "upper_95": upper.values})
                fig, ax = plt.subplots(figsize=(13, 5))
                ax.plot(self.ts.index, self.ts.values, label="Actual", color="steelblue")
                ax.plot(idx, mean.values, label="Prophet Forecast",
                        color="red", linestyle="--")
                ax.fill_between(idx, lower.values, upper.values,
                                alpha=0.2, color="red", label="95% CI")
                ax.set_title(f"Prophet Forecast  [{freq}]", fontsize=13)
            else:
                order, seasonal_order = self.get_orders()
                result = SARIMAX(self.ts, order=order, seasonal_order=seasonal_order,
                                 enforce_stationarity=False,
                                 enforce_invertibility=False).fit(disp=False)
                fc   = result.get_forecast(steps=h)
                mean = fc.predicted_mean; ci = fc.conf_int()
                self.forecast_result = pd.DataFrame({
                    "date": mean.index, "forecast": mean.values,
                    "lower_95": ci.iloc[:, 0].values, "upper_95": ci.iloc[:, 1].values})
                p, d, q = self.p_var.get(), self.d_var.get(), self.q_var.get()
                fig, ax = plt.subplots(figsize=(13, 5))
                ax.plot(self.ts.index, self.ts.values, label="Actual", color="steelblue")
                ax.plot(mean.index, mean.values, label="Forecast",
                        color="red", linestyle="--")
                ax.fill_between(mean.index, ci.iloc[:, 0], ci.iloc[:, 1],
                                alpha=0.2, color="red", label="95% CI")
                ax.set_title(
                    f"{self.model_var.get()} Forecast  [{freq}]  ({p},{d},{q})",
                    fontsize=13)
            ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout(); plt.show()
            messagebox.showinfo("Done",
                f"Forecast complete for {h} {freq} periods.\nClick 'Save CSV' to export.")
        except Exception as e:
            messagebox.showerror("Model Error", str(e))

    def save_csv(self):
        if self.forecast_result is None:
            messagebox.showwarning("Warning", "Run forecast first."); return
        fp = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if fp:
            self.forecast_result.to_csv(fp, index=False)
            messagebox.showinfo("Saved", f"Forecast saved:\n{fp}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ForecastApp(root)
    root.mainloop()