import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from tkinter.filedialog import askopenfilename
from ttkthemes import ThemedTk
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
import threading

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve
)
from imblearn.pipeline import Pipeline

import shap

class FraudDetectionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Sukčiavimo aptikimo sistema")
        self.master.geometry("1400x900")
        self.csv_file_path      = None
        self.use_default_params = tk.BooleanVar(value=True)
        self.show_cm_var        = tk.BooleanVar(value=True)
        self.show_shap_var      = tk.BooleanVar(value=True)
        self.use_cv             = tk.BooleanVar(value=False)
        self.param_values   = {}
        self.param_defaults = {}
        self.param_entries  = {}
        self.features       = None
        self.metrics_labels = {}
        self.pred_metrics_labels = {}
        self.trained_pipe   = None
        self.setup_ui()

    def setup_ui(self):
        header = ttk.Frame(self.master)
        header.pack(fill=tk.X, pady=10)
        ttk.Label(header, text="Sukčiavimo aptikimo sistema", font=("Arial",18,"bold")).pack(side=tk.LEFT)

        main = ttk.Panedwindow(self.master, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        input_frame = ttk.Labelframe(main, text="Įvestis", padding=15)
        main.add(input_frame, weight=1)

        ttk.Button(input_frame, text="Pasirinkti CSV failą", command=self.select_file)\
            .grid(row=0, column=0, columnspan=2, pady=(0,5), sticky="ew")
        self.csv_label = ttk.Label(input_frame, text="Nėra pasirinkto failo", wraplength=200)
        self.csv_label.grid(row=1, column=0, columnspan=2, pady=(0,10), sticky="w")

        ttk.Button(input_frame, text="Atlikti duomenų analizę", command=self.start_data_analysis_thread)\
            .grid(row=2, column=0, columnspan=2, pady=(0,10), sticky="ew")

        ttk.Label(input_frame, text="Pasirinkite algoritmą:")\
            .grid(row=3, column=0, columnspan=2, sticky="w")
        self.algorithm_var = tk.StringVar()
        algo_cb = ttk.Combobox(
            input_frame, textvariable=self.algorithm_var,
            values=[
                "Logistinė regresija","Sprendimų medis","Atsitiktinis miškas",
                "Gradientinis stiprinimas","Neuroninis tinklas","MLP eksperimentai"
            ], state="readonly"
        )
        algo_cb.grid(row=4, column=0, columnspan=2, pady=5, sticky="ew")
        algo_cb.current(0)
        algo_cb.bind("<<ComboboxSelected>>", lambda _: self.update_parameters())

        self.params_frame = ttk.Labelframe(input_frame, text="Parametrai", padding=10)
        self.params_frame.grid(row=5, column=0, columnspan=2, pady=10, sticky="ew")
        self.update_parameters()

        ttk.Checkbutton(input_frame, text="Rodyti Confusion Matrix", variable=self.show_cm_var)\
            .grid(row=6, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(input_frame, text="Rodyti SHAP Summary", variable=self.show_shap_var)\
            .grid(row=7, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(input_frame, text="Naudoti 5-fold CV", variable=self.use_cv)\
            .grid(row=8, column=0, columnspan=2, pady=(0,10), sticky="w")

        ttk.Button(input_frame, text="Apmokyti modelį", command=self.start_training_thread)\
            .grid(row=9, column=0, columnspan=2, pady=(0,10), sticky="ew")
        self.progress = ttk.Progressbar(input_frame, orient="horizontal", mode="determinate", maximum=100)
        self.progress.grid(row=10, column=0, columnspan=2, sticky="ew")
        self.status_label = ttk.Label(input_frame, text="Būsena: pasiruošta")
        self.status_label.grid(row=11, column=0, columnspan=2, pady=(5,0), sticky="w")

        ttk.Button(input_frame, text="Nuspėti naujas klases", command=self.start_prediction_thread)\
            .grid(row=12, column=0, columnspan=2, pady=(10,5), sticky="ew")

        output_frame = ttk.Labelframe(main, text="Rezultatai", padding=15)
        main.add(output_frame, weight=3)

        labels = [
            "Tikslumas","Preciziškumas","Recall","F1-rodiklis","ROC AUC","PR AUC",
            "Prec_kls0","Rec_kls0","F1_kls0","Prec_kls1","Rec_kls1","F1_kls1"
        ]

        ttk.Label(output_frame, text="Metrikos").grid(row=0, column=0, sticky="w")
        ttk.Label(output_frame, text="Testavimo").grid(row=0, column=1, sticky="w")
        ttk.Label(output_frame, text="Spėjimo").grid(row=0, column=2, sticky="w")

        for i, key in enumerate(labels, start=1):
            ttk.Label(output_frame, text=f"{key}:").grid(row=i, column=0, sticky="w", pady=2)
            lbl = ttk.Label(output_frame, text="--", font=("Arial",12,"bold"))
            lbl.grid(row=i, column=1, sticky="w", pady=2)
            self.metrics_labels[key] = lbl

            plbl = ttk.Label(output_frame, text="--", font=("Arial",12,"bold"))
            plbl.grid(row=i, column=2, sticky="w", pady=2)
            self.pred_metrics_labels[key] = plbl

    def select_file(self):
        path = askopenfilename(filetypes=[("CSV failai","*.csv")])
        if path:
            self.csv_file_path = path
            self.csv_label.config(text=path)

    def start_data_analysis_thread(self):
        threading.Thread(target=self.analyze_data, daemon=True).start()

    def start_training_thread(self):
        threading.Thread(target=self.train_and_evaluate_model, daemon=True).start()

    def start_prediction_thread(self):
        threading.Thread(target=self.predict_new, daemon=True).start()

    def analyze_data(self):
        if not self.csv_file_path:
            messagebox.showerror("Klaida","Pasirinkite CSV failą prieš analizę.")
            return
        df = pd.read_csv(self.csv_file_path)
        report = [
            f"Eilučių skaičius: {len(df)}",
            f"Stulpelių skaičius: {len(df.columns)}",
            f"Null reikšmių: {df.isnull().sum().sum()}",
            f"NaN reikšmių: {df.isna().sum().sum()}",
            f"Pasikartojančių eilučių: {df.duplicated().sum()}"
        ]
        report.append("Stulpelių unikalios reikšmės:")
        for col in df.columns:
            report.append(f"  {col}: {df[col].nunique()} unikalių")
        self.master.after(0, self.show_analysis, df, report)

    def show_analysis(self, df, report):
        win = tk.Toplevel(self.master); win.title("Duomenų analizės ataskaita")
        text = scrolledtext.ScrolledText(win, width=80, height=20)
        text.pack(fill=tk.BOTH, expand=True)
        text.insert(tk.END, "\n".join(report)); text.config(state=tk.DISABLED)
        if 'Class' in df.columns:
            fig, ax = plt.subplots(figsize=(6,4))
            df['Class'].value_counts().plot.bar(ax=ax)
            ax.set(xlabel="Klasė", ylabel="Kiekis", title="Klasių pasiskirstymas")
            self.display_plot(fig, "Klasių pasiskirstymas")
        if 'Time' in df.columns:
            fig, ax = plt.subplots(figsize=(6,4))
            df['Time'].hist(bins=30, ax=ax)
            ax.set(xlabel="Laikas", ylabel="Kiekis", title="Laiko histograma")
            self.display_plot(fig, "Laiko histograma")
        corr = df.select_dtypes(include=np.number).corr()
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(corr, cmap='coolwarm', ax=ax)
        ax.set_title("Koreliacijos matrica")
        self.display_plot(fig, "Koreliacijos matrica")

    def update_parameters(self):
        for w in self.params_frame.winfo_children(): w.destroy()
        ttk.Checkbutton(self.params_frame, text="Numatyti parametrai",
                        variable=self.use_default_params,
                        command=self.toggle_parameter_inputs).pack(anchor="w", pady=(0,5))
        algo = self.algorithm_var.get()
        self.param_values.clear(); self.param_defaults.clear(); self.param_entries.clear()
        def add_param(name, default):
            frm = ttk.Frame(self.params_frame); frm.pack(fill="x", pady=2)
            ttk.Label(frm, text=f"{name}:").pack(side="left")
            var = tk.StringVar(value=default)
            ent = ttk.Entry(frm, textvariable=var)
            ent.pack(side="right", fill="x", expand=True)
            ent.config(state="disabled" if self.use_default_params.get() else "normal")
            self.param_values[name]=var; self.param_defaults[name]=default; self.param_entries[name]=ent

        if algo=="Logistinė regresija":
            add_param("C","1.0"); add_param("penalty","l2"); add_param("solver","lbfgs"); add_param("max_iter","1000")
        elif algo=="Sprendimų medis":
            add_param("criterion","gini"); add_param("splitter","best"); add_param("max_depth","None"); add_param("min_samples_split","2")
        elif algo=="Atsitiktinis miškas":
            add_param("n_estimators","100"); add_param("criterion","gini"); add_param("max_features","auto"); add_param("max_depth","None")
        elif algo=="Gradientinis stiprinimas":
            add_param("learning_rate","0.1"); add_param("n_estimators","100"); add_param("max_depth","3"); add_param("subsample","1.0")
        elif algo=="Neuroninis tinklas":
            add_param("hidden_layer_sizes","100"); add_param("activation","relu"); add_param("solver","adam")
            add_param("alpha","0.0001"); add_param("batch_size","auto"); add_param("learning_rate_init","0.001")
            add_param("max_iter","200"); add_param("early_stopping","False")
        elif algo=="MLP eksperimentai":
            ttk.Label(self.params_frame, text="Nėra parametrų šiam algoritmui").pack()

    def toggle_parameter_inputs(self):
        for name, ent in self.param_entries.items():
            if self.use_default_params.get():
                ent.config(state="disabled"); self.param_values[name].set(self.param_defaults[name])
            else:
                ent.config(state="normal")

    def resource_path(self, rel):
        import sys, os
        base = getattr(sys, '_MEIPASS', os.path.dirname(__file__))
        return os.path.join(base, rel)

    def display_plot(self, fig, title):
        win = tk.Toplevel(self.master); win.title(title)
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_progress(self, val):
        self.master.after(0, lambda: self.progress.config(value=val))

    def create_model(self):
        p = self.param_values; algo = self.algorithm_var.get()
        if algo=="Logistinė regresija":
            return LogisticRegression(C=float(p["C"].get()), penalty=p["penalty"].get(),
                                      solver=p["solver"].get(), max_iter=int(p["max_iter"].get()))
        if algo=="Sprendimų medis":
            md=None if p["max_depth"].get()=="None" else int(p["max_depth"].get())
            return DecisionTreeClassifier(criterion=p["criterion"].get(),
                                          splitter=p["splitter"].get(),
                                          max_depth=md,
                                          min_samples_split=int(p["min_samples_split"].get()))
        if algo=="Atsitiktinis miškas":
            md=None if p["max_depth"].get()=="None" else int(p["max_depth"].get())
            return RandomForestClassifier(n_estimators=int(p["n_estimators"].get()),
                                          criterion=p["criterion"].get(),
                                          max_features=p["max_features"].get(),
                                          max_depth=md)
        if algo=="Gradientinis stiprinimas":
            return GradientBoostingClassifier(learning_rate=float(p["learning_rate"].get()),
                                              n_estimators=int(p["n_estimators"].get()),
                                              max_depth=int(p["max_depth"].get()),
                                              subsample=float(p["subsample"].get()))
        if algo=="Neuroninis tinklas":
            hl=tuple(map(int,p["hidden_layer_sizes"].get().split(",")))
            return MLPClassifier(hidden_layer_sizes=hl,
                                 activation=p["activation"].get(),
                                 solver=p["solver"].get(),
                                 alpha=float(p["alpha"].get()),
                                 batch_size=p["batch_size"].get(),
                                 learning_rate_init=float(p["learning_rate_init"].get()),
                                 max_iter=int(p["max_iter"].get()),
                                 early_stopping=(p["early_stopping"].get()=="True"))

    def run_mlp_experiments(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2
        )

        hidden_layers_list = [(50,), (100,), (50,50)]
        learning_rates    = [0.001, 0.01]
        early_stoppings   = [False, True]
        max_iters         = [300, 500, 800]

        total = (
            len(hidden_layers_list)
            * len(learning_rates)
            * len(early_stoppings)
            * len(max_iters)
        )
        results = []
        idx = 0

        for hl in hidden_layers_list:
            for lr in learning_rates:
                for es in early_stoppings:
                    for mi in max_iters:
                        m = MLPClassifier(
                            hidden_layer_sizes=hl,
                            learning_rate_init=lr,
                            early_stopping=es,
                            max_iter=mi
                        )
                        pipe = Pipeline([('model', m)])
                        pipe.fit(X_train, y_train)
                        preds = pipe.predict(X_test)

                        rec = recall_score(y_test, preds, zero_division=0)
                        f1  = f1_score(y_test, preds, zero_division=0)

                        results.append({
                            "hidden_layer_sizes": str(hl),
                            "learning_rate_init": lr,
                            "early_stopping": es,
                            "max_iter": mi,
                            "recall": rec,
                            "f1": f1
                        })

                        idx += 1
                        self.update_progress(int(idx/total*100))

        self.update_progress(100)

        df_res = pd.DataFrame(results)
        win = tk.Toplevel(self.master)
        win.title("MLP Eksperimentų Rezultatai")
        text = scrolledtext.ScrolledText(win, width=80, height=20)
        text.pack(fill=tk.BOTH, expand=True)
        text.insert(tk.END, df_res.to_string(index=False))
        text.config(state=tk.DISABLED)
        df_res.to_csv("mlp_experiment_results.csv", index=False)

    def train_and_evaluate_model(self):
        if not self.csv_file_path:
            messagebox.showerror("Klaida","Pasirinkite CSV failą prieš apmokymą"); return
        df = pd.read_csv(self.csv_file_path).sample(frac=1).reset_index(drop=True)
        X = df.drop(['Class','id'],axis=1,errors='ignore'); y = df['Class']
        self.features = X.columns; self.progress.config(value=0)
        if self.algorithm_var.get()=="MLP eksperimentai":
            self.run_mlp_experiments(X, y)
            return
        if self.use_cv.get():
            cv=StratifiedKFold(n_splits=5,shuffle=True)
            accs,precs,recs,f1s,rocs,prs=[],[],[],[],[],[]
            for i,(tr,te) in enumerate(cv.split(X,y)):
                m=self.create_model(); pipe=Pipeline([('model',m)])
                pipe.fit(X.iloc[tr],y.iloc[tr]); y_pred=pipe.predict(X.iloc[te])
                accs.append(accuracy_score(y.iloc[te],y_pred))
                rpt=classification_report(y.iloc[te],y_pred,output_dict=True)
                precs.append(rpt['1']['precision']); recs.append(rpt['1']['recall']); f1s.append(rpt['1']['f1-score'])
                if hasattr(m,'predict_proba'):
                    s=pipe.predict_proba(X.iloc[te])[:,1]
                    fpr,tpr,_=roc_curve(y.iloc[te],s); rocs.append(auc(fpr,tpr))
                    pr,rc,_=precision_recall_curve(y.iloc[te],s); prs.append(auc(rc,pr))
                self.update_progress(int((i+1)/5*100))
                last_pipe, last_te, last_pred, last_model = pipe, te, y_pred, m
            self.trained_pipe = last_pipe
            self.metrics_labels['Tikslumas'].config(text=f"{np.mean(accs):.4f}")
            self.metrics_labels['Preciziškumas'].config(text=f"{np.mean(precs):.4f}")
            self.metrics_labels['Recall'].config(text=f"{np.mean(recs):.4f}")
            self.metrics_labels['F1-rodiklis'].config(text=f"{np.mean(f1s):.4f}")
            self.metrics_labels['ROC AUC'].config(text=f"{np.mean(rocs):.4f}" if rocs else "--")
            self.metrics_labels['PR AUC'].config(text=f"{np.mean(prs):.4f}" if prs else "--")
            for k in ['Prec_kls0','Rec_kls0','F1_kls0','Prec_kls1','Rec_kls1','F1_kls1']:
                self.metrics_labels[k].config(text="--")
            self.status_label.config(text="5-fold CV baigta")
            if self.show_cm_var.get():
                cm=confusion_matrix(y.iloc[last_te], last_pred)
                self.display_plot(self.build_confusion_matrix(cm),"Confusion Matrix")
            if self.show_shap_var.get() and hasattr(last_model,'predict_proba'):
                bg=shap.sample(X.iloc[last_te],100)
                expl = shap.TreeExplainer(last_model,bg) if isinstance(last_model,(DecisionTreeClassifier,RandomForestClassifier,GradientBoostingClassifier)) else shap.KernelExplainer(last_model.predict_proba,bg)
                sample=X.iloc[last_te].sample(n=min(100,len(last_te)))
                sv=expl.shap_values(sample)
                if isinstance(sv,list): sv=sv[1]
                self.display_plot(self.build_shap_summary(sv,sample),"SHAP Summary")
            return

        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
        for i in range(10): threading.Event().wait(0.1); self.update_progress((i+1)*5)
        try:
            m=self.create_model(); pipe=Pipeline([('model',m)])
            pipe.fit(X_train,y_train); self.trained_pipe=pipe
            self.progress.config(value=50); self.status_label.config(text="Apmokyta")
            y_pred=pipe.predict(X_test); rpt=classification_report(y_test,y_pred,output_dict=True)
            acc=accuracy_score(y_test,y_pred)
            prec=rpt['1']['precision']; rec=rpt['1']['recall']; f1=rpt['1']['f1-score']
            self.metrics_labels['Tikslumas'].config(text=f"{acc:.4f}")
            self.metrics_labels['Preciziškumas'].config(text=f"{prec:.4f}")
            self.metrics_labels['Recall'].config(text=f"{rec:.4f}")
            self.metrics_labels['F1-rodiklis'].config(text=f"{f1:.4f}")
            for cls in ['0','1']:
                self.metrics_labels[f'Prec_kls{cls}'].config(text=f"{rpt[cls]['precision']:.4f}")
                self.metrics_labels[f'Rec_kls{cls}'].config(text=f"{rpt[cls]['recall']:.4f}")
                self.metrics_labels[f'F1_kls{cls}'].config(text=f"{rpt[cls]['f1-score']:.4f}")
            if hasattr(m,'predict_proba'):
                s=pipe.predict_proba(X_test)[:,1]
                fpr,tpr,_=roc_curve(y_test,s); roc_val=auc(fpr,tpr)
                pr,rc,_=precision_recall_curve(y_test,s); pr_val=auc(rc,pr)
                self.metrics_labels['ROC AUC'].config(text=f"{roc_val:.4f}")
                self.metrics_labels['PR AUC'].config(text=f"{pr_val:.4f}")
            else:
                self.metrics_labels['ROC AUC'].config(text="--"); self.metrics_labels['PR AUC'].config(text="--")
            if self.show_cm_var.get():
                cm=confusion_matrix(y_test,y_pred)
                self.display_plot(self.build_confusion_matrix(cm),"Confusion Matrix")
            if self.show_shap_var.get() and hasattr(m,'predict_proba'):
                bg=shap.sample(X_train,100)
                expl = shap.TreeExplainer(m,bg) if isinstance(m,(DecisionTreeClassifier,RandomForestClassifier,GradientBoostingClassifier)) else shap.KernelExplainer(m.predict_proba,bg)
                sample=X_test.sample(n=min(100,len(X_test)))
                sv=expl.shap_values(sample)
                if isinstance(sv,list): sv=sv[1]
                self.display_plot(self.build_shap_summary(sv,sample),"SHAP Summary")
            self.update_progress(100)
        except Exception as e:
            messagebox.showerror("Apmokymo klaida",f"Įvyko klaida: {e}")
            self.status_label.config(text="Būsena: klaida apmokant"); self.update_progress(0)

    def predict_new(self):
        path = askopenfilename(filetypes=[("CSV failai","*.csv")])
        if not path:
            return
        df_new = pd.read_csv(path).dropna(subset=['Class'])
        self.progress.config(value=0); self.update_progress(10)
        X_new = df_new.drop(['Class','id'],axis=1,errors='ignore').fillna(method='ffill')
        y_true = df_new['Class']
        self.update_progress(30)
        preds = self.trained_pipe.predict(X_new)
        self.update_progress(50)
        acc   = accuracy_score(y_true, preds)
        prec  = precision_score(y_true, preds, zero_division=0)
        rec   = recall_score(y_true, preds, zero_division=0)
        f1    = f1_score(y_true, preds, zero_division=0)
        if hasattr(self.trained_pipe.named_steps['model'], 'predict_proba'):
            probs = self.trained_pipe.predict_proba(X_new)[:,1]
            fpr, tpr, _ = roc_curve(y_true, probs)
            roc_val = auc(fpr, tpr)
            pr_vals, rc_vals, _ = precision_recall_curve(y_true, probs)
            pr_val = auc(rc_vals, pr_vals)
        else:
            roc_val = pr_val = None
        self.update_progress(80)
        rpt = classification_report(y_true, preds, output_dict=True, zero_division=0)
        self.pred_metrics_labels['Tikslumas'].config(text=f"{acc:.4f}")
        self.pred_metrics_labels['Preciziškumas'].config(text=f"{prec:.4f}")
        self.pred_metrics_labels['Recall'].config(text=f"{rec:.4f}")
        self.pred_metrics_labels['F1-rodiklis'].config(text=f"{f1:.4f}")
        self.pred_metrics_labels['ROC AUC'].config(text=f"{roc_val:.4f}" if roc_val is not None else "--")
        self.pred_metrics_labels['PR AUC'].config(text=f"{pr_val:.4f}" if pr_val is not None else "--")
        for cls in ['0','1']:
            if str(cls) in rpt:
                cls_r = rpt[str(cls)]
                p = cls_r.get('precision',0.0)
                r = cls_r.get('recall',0.0)
                f = cls_r.get('f1-score',0.0)
                self.pred_metrics_labels[f'Prec_kls{cls}'].config(text=f"{p:.4f}")
                self.pred_metrics_labels[f'Rec_kls{cls}'].config(text=f"{r:.4f}")
                self.pred_metrics_labels[f'F1_kls{cls}'].config(text=f"{f:.4f}")
            else:
                self.pred_metrics_labels[f'Prec_kls{cls}'].config(text="--")
                self.pred_metrics_labels[f'Rec_kls{cls}'].config(text="--")
                self.pred_metrics_labels[f'F1_kls{cls}'].config(text="--")
        self.update_progress(100)

    def build_confusion_matrix(self, cm):
        fig = plt.figure(figsize=(5,4), dpi=100)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=fig.add_subplot())
        plt.title('Confusion Matrix')
        return fig

    def build_shap_summary(self, shap_vals, sample):
        shap_vals = np.nan_to_num(shap_vals, nan=0.0, posinf=0.0, neginf=0.0)
        shap.summary_plot(shap_vals, sample, show=False)
        return plt.gcf()

if __name__ == "__main__":
    root = ThemedTk(theme="radiance")
    style = ttk.Style(); style.theme_use("radiance")
    app = FraudDetectionApp(root)
    root.mainloop()
