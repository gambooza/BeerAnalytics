import pandas as pd
import customtkinter as ctk
import re
from tkinter import filedialog, messagebox
from sklearn.metrics import accuracy_score, cohen_kappa_score, recall_score, precision_score, f1_score

class AnalizadorProduccion(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Gambooza QA - Production Readiness Validator")
        self.geometry("1150x750")
        ctk.set_appearance_mode("dark")
        
        # --- UI ---
        self.label_titulo = ctk.CTkLabel(self, text="Validador de Calidad para Producción", font=("Arial", 22, "bold"))
        self.label_titulo.pack(pady=15)

        self.btn_real = ctk.CTkButton(self, text="1. Cargar CSV REAL", command=lambda: self.select_file('real'))
        self.btn_real.pack(pady=5)
        self.lbl_real = ctk.CTkLabel(self, text="...", text_color="gray")
        self.lbl_real.pack()

        self.btn_pred = ctk.CTkButton(self, text="2. Cargar CSV CALCULADO", command=lambda: self.select_file('pred'), fg_color="#34495e")
        self.btn_pred.pack(pady=5)
        self.lbl_pred = ctk.CTkLabel(self, text="...", text_color="gray")
        self.lbl_pred.pack()

        self.btn_run = ctk.CTkButton(self, text="ANALIZAR PARA PRODUCCIÓN", command=self.calcular, fg_color="#27ae60", height=45)
        self.btn_run.pack(pady=20)

        self.text_area = ctk.CTkTextbox(self, font=("Consolas", 11), width=1100, height=400)
        self.text_area.pack(padx=20, pady=10)

        # Configuración de colores para el veredicto (Sin fuentes especiales para evitar errores)
        self.text_area.tag_config("green", foreground="#2ecc71")
        self.text_area.tag_config("yellow", foreground="#f1c40f")
        self.text_area.tag_config("red", foreground="#e74c3c")

        self.f_real = ""
        self.f_pred = ""

    def select_file(self, tipo):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if path:
            if tipo == 'real': 
                self.f_real = path
                self.lbl_real.configure(text=path.split("/")[-1])
            else: 
                self.f_pred = path
                self.lbl_pred.configure(text=path.split("/")[-1])

    def procesar_identificador(self, texto):
        texto_str = str(texto).strip()
        match = re.search(r'\d+', texto_str)
        return f"TAP {match.group()}" if match else texto_str

    def calcular(self):
        if not self.f_real or not self.f_pred:
            messagebox.showerror("Error", "Carga ambos archivos.")
            return

        try:
            # Lectura de archivos
            df_real = pd.read_csv(self.f_real, sep=';')
            df_pred = pd.read_csv(self.f_pred, sep=';')

            df_real['Grupo_ID'] = df_real['Grifo'].apply(self.procesar_identificador)
            df_pred['Grupo_ID'] = df_pred['Grifo'].apply(self.procesar_identificador)

            # Cruce de datos
            comparativa = pd.merge(
                df_real[['ID Único', 'Grupo_ID', 'Cervezas']], 
                df_pred[['ID Único', 'Cervezas']], 
                on='ID Único', how='left', suffixes=('_R', '_P')
            ).fillna(0)

            # Lógica binaria: ¿Hubo apertura?
            comparativa['Detec_R'] = (comparativa['Cervezas_R'] > 0).astype(int)
            comparativa['Detec_P'] = (comparativa['Cervezas_P'] > 0).astype(int)

            self.text_area.delete("1.0", "end")
            header = f"{'SURTIDOR':<15} | {'ACC_CONTEO':<12} | {'KAPPA':<10} | {'SENSIV (Rec)':<12} | {'PRECISION':<12} | {'F1-SCORE':<10}\n"
            self.text_area.insert("end", header + "-"*105 + "\n")

            for g_id, g in comparativa.groupby('Grupo_ID'):
                acc_c = accuracy_score(g['Cervezas_R'], g['Cervezas_P'])
                try: kap_c = cohen_kappa_score(g['Cervezas_R'], g['Cervezas_P'])
                except: kap_c = 1.0 if acc_c == 1.0 else 0.0

                sens = recall_score(g['Detec_R'], g['Detec_P'], zero_division=0)
                prec = precision_score(g['Detec_R'], g['Detec_P'], zero_division=0)
                f1 = f1_score(g['Detec_R'], g['Detec_P'], zero_division=0)

                linea = f"{g_id:<15} | {acc_c:<12.2%} | {kap_c:<10.4f} | {sens:<12.2%} | {prec:<12.2%} | {f1:<10.2%}\n"
                self.text_area.insert("end", linea)

            # MÉTRICAS TOTALES
            acc_gl = accuracy_score(comparativa['Cervezas_R'], comparativa['Cervezas_P'])
            sens_gl = recall_score(comparativa['Detec_R'], comparativa['Detec_P'], zero_division=0)
            prec_gl = precision_score(comparativa['Detec_R'], comparativa['Detec_P'], zero_division=0)
            
            self.text_area.insert("end", "="*105 + "\n")
            self.text_area.insert("end", f"{'TOTAL GLOBAL':<15} | {acc_gl:<12.2%} | {'-':<10} | {sens_gl:<12.2%} | {prec_gl:<12.2%} | {'-':<10}\n\n")

            # --- VEREDICTO FINAL ---
            self.text_area.insert("end", "📢 VEREDICTO FINAL:\n")
            
            if sens_gl >= 0.90 and acc_gl >= 0.80:
                self.text_area.insert("end", "✅ [MODELO ACEPTABLE PARA PRODUCCIÓN]\n", "green")
                self.text_area.insert("end", "El modelo detecta casi todas las aperturas y tiene un error de conteo bajo.\n")
            elif sens_gl < 0.90 and sens_gl >= 0.75:
                self.text_area.insert("end", "⚠️ [MODELO EN REVISIÓN]\n", "yellow")
                self.text_area.insert("end", "La detección es buena pero se pierde más del 10% de los eventos reales.\n")
            else:
                self.text_area.insert("end", "❌ [MODELO NO ACEPTABLE]\n", "red")
                self.text_area.insert("end", "La sensibilidad es demasiado baja. El modelo se salta demasiados eventos reales.\n")

        except Exception as e:
            messagebox.showerror("Error", f"Error en el proceso: {str(e)}")

if __name__ == "__main__":
    app = AnalizadorProduccion()
    app.mainloop()