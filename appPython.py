import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime, timedelta
import random
import re

class ProjectMasterUltra:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ProjectMaster Ultra | Ingeniería y Gestión Pro")
        self.root.geometry("1450x850")
        
        self.df = pd.DataFrame()
        self.orden = []
        self.unidad_predominante = "Días"
        
        self.colores = {
            "bg": "#1e1e2e", "side": "#181825", "accent": "#89b4fa",
            "text": "#cdd6f4", "critico": "#f38ba8", "normal": "#94e2d5",
            "borde": "#45475a"
        }
        
        self.configurar_estilos()
        self.crear_layout()

    def configurar_estilos(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background="#313244", foreground="white", fieldbackground="#313244", rowheight=30)
        style.map("Treeview", background=[('selected', '#89b4fa')])
        style.configure("Side.TFrame", background=self.colores["side"])
        style.configure("TLabel", background=self.colores["bg"], foreground=self.colores["text"])
        style.configure("TButton", padding=5)

    def crear_layout(self):
        # Sidebar - Control de mandos
        sidebar = ttk.Frame(self.root, style="Side.TFrame", width=320)
        sidebar.pack(side="left", fill="y")
        
        ttk.Label(sidebar, text="ENGINEERING PRO", font=("Impact", 24), foreground=self.colores["accent"], background=self.colores["side"]).pack(pady=20)
        
        config_frame = ttk.LabelFrame(sidebar, text=" Configuración de Proyecto ", padding=10)
        config_frame.pack(pady=5, padx=15, fill="x")

        ttk.Label(config_frame, text="Fecha Inicio (dd/mm/aaaa):", background=self.colores["side"]).pack(anchor="w")
        self.fecha_inicio_ent = ttk.Entry(config_frame)
        self.fecha_inicio_ent.insert(0, datetime.now().strftime("%d/%m/%Y"))
        self.fecha_inicio_ent.pack(fill="x", pady=5)

        ttk.Label(config_frame, text="Calendario Laboral:", background=self.colores["side"]).pack(anchor="w")
        self.jornada_cb = ttk.Combobox(config_frame, values=["Lunes a Viernes", "Lunes a Sábado", "Lunes a Domingo"])
        self.jornada_cb.set("Lunes a Viernes")
        self.jornada_cb.pack(fill="x", pady=5)

        # Botonera Principal
        btns = [
            ("📁 Cargar Excel", self.cargar_excel),
            ("⚙️ Calcular Ruta Crítica", self.ejecutar_cpm_completo),
            ("📊 Ver Gantt Pro", self.mostrar_gantt),
            ("🕸️ Red de Actividades", self.mostrar_red),
            ("🎲 Simulación Monte Carlo", self.monte_carlo_pro),
            ("💾 Guardar Proyecto", self.guardar_proyecto)
        ]
        
        for texto, comando in btns:
            ttk.Button(sidebar, text=texto, command=comando).pack(pady=6, padx=20, fill="x")

        # Área de Datos
        self.main_area = ttk.Frame(self.root)
        self.main_area.pack(side="right", fill="both", expand=True, padx=20, pady=20)
        
        self.columnas_tabla = ("Actividad", "Duracion", "ES", "EF", "LS", "LF", "Holgura", "Critica")
        self.tree = ttk.Treeview(self.main_area, columns=self.columnas_tabla, show="headings")
        for col in self.columnas_tabla:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor="center")
        self.tree.pack(fill="both", expand=True)

    # =====================================================
    # LÓGICA DE INGENIERÍA: CPM Y GRAFOS
    # =====================================================
    def limpiar_predecesores(self, valor):
        if pd.isna(valor) or str(valor).strip() == "" or str(valor).lower() in ["nan", "none", "-"]:
            return []
        return [p.strip().upper() for p in str(valor).split(",") if p.strip()]

    def detectar_ciclos_y_ordenar(self):
        # Algoritmo de Kahn para Orden Topológico y Detección de Ciclos
        nodos = self.df["Actividad"].astype(str).str.upper().tolist()
        adj = {n: [] for n in nodos}
        grados_entrada = {n: 0 for n in nodos}
        
        for _, row in self.df.iterrows():
            u = str(row["Actividad"]).upper()
            preds = self.limpiar_predecesores(row.get("Predecesor", ""))
            for p in preds:
                if p in adj:
                    adj[p].append(u)
                    grados_entrada[u] += 1
        
        cola = [n for n in nodos if grados_entrada[n] == 0]
        orden = []
        
        while cola:
            u = cola.pop(0)
            orden.append(u)
            for v in adj[u]:
                grados_entrada[v] -= 1
                if grados_entrada[v] == 0:
                    cola.append(v)
        
        if len(orden) != len(nodos):
            return None # Hay un ciclo (Error de ingeniería)
        return orden

    def ejecutar_cpm_completo(self):
        if self.df.empty:
            messagebox.showwarning("Aviso", "Cargue un archivo primero.")
            return
        
        try:
            self.df["Dias_Num"] = self.df["Duracion"].apply(self.convertir_a_dias)
            self.orden = self.detectar_ciclos_y_ordenar()
            
            if self.orden is None:
                messagebox.showerror("Error de Ciclo", "¡Error Crítico! Se detectó una dependencia circular (ej: A -> B -> A). Revise sus predecesores.")
                return

            # Forward Pass
            es, ef = {}, {}
            for act in self.orden:
                row = self.df[self.df["Actividad"].astype(str).str.upper() == act].iloc[0]
                preds = self.limpiar_predecesores(row.get("Predecesor", ""))
                es[act] = max([ef[p] for p in preds if p in ef], default=0)
                ef[act] = es[act] + row["Dias_Num"]

            # Backward Pass
            max_dur = max(ef.values(), default=0)
            ls, lf = {}, {}
            for act in reversed(self.orden):
                sucesores = [h for h, r in self.df.iterrows() if act in self.limpiar_predecesores(r.get("Predecesor", ""))]
                sucesores_act = [str(self.df.at[h, "Actividad"]).upper() for h in sucesores]
                lf[act] = min([ls[s] for s in sucesores_act if s in ls], default=max_dur)
                ls[act] = lf[act] - self.df[self.df["Actividad"].astype(str).str.upper() == act]["Dias_Num"].iloc[0]

            self.df["ES"] = self.df["Actividad"].str.upper().map(es)
            self.df["EF"] = self.df["Actividad"].str.upper().map(ef)
            self.df["LS"] = self.df["Actividad"].str.upper().map(ls)
            self.df["LF"] = self.df["Actividad"].str.upper().map(lf)
            self.df["Holgura"] = self.df["LS"] - self.df["ES"]
            self.df["Critica"] = self.df["Holgura"].apply(lambda x: "SÍ" if abs(x) < 0.001 else "NO")
            
            self.actualizar_tabla_ui()
            messagebox.showinfo("Cálculo Exitoso", f"Proyecto programado. Duración total: {max_dur} {self.unidad_predominante}")
        except Exception as e:
            messagebox.showerror("Error", f"Error en procesamiento: {str(e)}")

    # =====================================================
    # VISUALIZACIÓN RED CPM (MEJORADA)
    # =====================================================
    def mostrar_red(self):
        if "ES" not in self.df.columns: return
        
        fig = go.Figure()
        # Posicionamiento escalonado para evitar traslapes
        pos = {act: (self.df[self.df["Actividad"].str.upper()==act]["ES"].iloc[0] * 1.5, -i * 2.5) 
               for i, act in enumerate(self.orden)}

        # Dibujar conexiones (Flechas)
        for _, row in self.df.iterrows():
            hijo = str(row["Actividad"]).upper()
            preds = self.limpiar_predecesores(row.get("Predecesor", ""))
            for p in preds:
                if p in pos:
                    es_critico = row["Critica"] == "SÍ" and self.df[self.df["Actividad"].str.upper()==p]["Critica"].iloc[0] == "SÍ"
                    fig.add_trace(go.Scatter(
                        x=[pos[p][0], pos[hijo][0]], y=[pos[p][1], pos[hijo][1]],
                        line=dict(color=self.colores["critico"] if es_critico else "#444", width=2 if es_critico else 1),
                        hoverinfo='none', mode='lines+markers', marker=dict(symbol="arrow-right", size=10)
                    ))

        # Dibujar Nodos (Cajas de Ingeniería)
        for act, (x, y) in pos.items():
            r = self.df[self.df["Actividad"].str.upper() == act].iloc[0]
            color_nodo = self.colores["critico"] if r["Critica"] == "SÍ" else self.colores["accent"]
            
            # El cuerpo del nodo
            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode="markers",
                marker=dict(size=45, color=color_nodo, symbol="square"),
                hoverinfo="text", hovertext=f"Actividad: {act}<br>Holgura: {r.Holgura}"
            ))
            
            # Texto dentro y debajo para orden
            fig.add_annotation(x=x, y=y, text=f"<b>{act}</b>", showarrow=False, font=dict(color="white"))
            fig.add_annotation(
                x=x, y=y-1.1, 
                text=f"ES: {int(r.ES)} | EF: {int(r.EF)}<br>LS: {int(r.LS)} | LF: {int(r.LF)}",
                showarrow=False, bordercolor=color_nodo, bgcolor="#222", font=dict(size=10, color="white")
            )

        # Glosario en la esquina superior derecha
        glosario = ("<b>REFERENCIAS</b><br>"
                    "ES: Inicio Temprano<br>"
                    "EF: Fin Temprano<br>"
                    "LS: Inicio Tardío<br>"
                    "LF: Fin Tardío<br>"
                    "Holgura: Tiempo de flexibilidad")
        
        fig.add_annotation(
            xref="paper", yref="paper", x=0.98, y=0.98,
            text=glosario, showarrow=False, align="left",
            bgcolor="rgba(255,255,255,0.8)", bordercolor="black", borderpad=4
        )

        fig.update_layout(title="Red de Precedencias CPM", plot_bgcolor="#111", showlegend=False)
        fig.show()

    # =====================================================
    # SIMULACIÓN MONTE CARLO (INGENIERÍA REAL)
    # =====================================================
    def monte_carlo_pro(self):
        if "ES" not in self.df.columns: return
        
        iteraciones = 2500
        resultados = []
        
        # Simulación de variabilidad por actividad
        for _ in range(iteraciones):
            tiempos_sim = {}
            for act in self.orden:
                row = self.df[self.df["Actividad"].str.upper()==act].iloc[0]
                base = row["Dias_Num"]
                # Distribución Triangular: Optimista (90%), Más probable (100%), Pesimista (140%)
                d_sim = random.triangular(base * 0.9, base * 1.4, base)
                
                preds = self.limpiar_predecesores(row.get("Predecesor", ""))
                es_v = max([tiempos_sim[p] for p in preds if p in tiempos_sim], default=0)
                tiempos_sim[act] = es_v + d_sim
            resultados.append(max(tiempos_sim.values()))

        # Gráfico Estadístico
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=resultados, nbinsx=40, name="Frecuencia",
            marker=dict(color=self.colores["accent"], line=dict(color="white", width=1))
        ))
        
        prob_95 = np.percentile(resultados, 95)
        media = np.mean(resultados)

        fig.add_vline(x=prob_95, line_dash="dash", line_color=self.colores["critico"], 
                     annotation_text=f"95% Seguridad: {prob_95:.1f}")
        
        fig.update_layout(
            title=f"Análisis de Riesgo Monte Carlo - Finalización ({self.unidad_predominante})",
            xaxis_title="Días Totales Estimados",
            yaxis_title="Cantidad de Escenarios",
            template="plotly_white"
        )
        fig.show()

    # =====================================================
    # UTILIDADES ADICIONALES
    # =====================================================
    def convertir_a_dias(self, valor):
        s = str(valor).lower()
        num = float(re.findall(r"\d+\.?\d*", s)[0]) if re.findall(r"\d+\.?\d*", s) else 0
        if "sem" in s: self.unidad_predominante = "Semanas"; return num * 5
        if "mes" in s: self.unidad_predominante = "Meses"; return num * 22
        return num

    def cargar_excel(self):
        ruta = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
        if ruta:
            temp = pd.read_excel(ruta)
            # Normalización inteligente de nombres de columna
            mapping = {"Actividad": ["tarea", "actividad", "proceso", "nombre"],
                       "Duracion": ["tiempo", "duracion", "dias", "días"],
                       "Predecesor": ["predecesor", "dependencia", "precedencia"]}
            
            new_df = pd.DataFrame()
            for key, targets in mapping.items():
                match = [c for c in temp.columns if c.lower() in targets]
                if match: new_df[key] = temp[match[0]]
            
            self.df = new_df
            self.actualizar_tabla_ui()

    def mostrar_gantt(self):
        if "ES" not in self.df.columns: return
        fb = datetime.strptime(self.fecha_inicio_ent.get(), "%d/%m/%Y")
        
        df_g = self.df.copy()
        df_g["Inicio"] = df_g["ES"].apply(lambda x: fb + timedelta(days=int(x)))
        df_g["Fin"] = df_g["EF"].apply(lambda x: fb + timedelta(days=int(x)))
        
        fig = px.timeline(df_g, x_start="Inicio", x_end="Fin", y="Actividad", color="Critica",
                         color_discrete_map={"SÍ": self.colores["critico"], "NO": self.colores["normal"]},
                         title="Cronograma Maestro de Obra (Gantt)")
        fig.update_yaxes(autorange="reversed")
        fig.show()

    def actualizar_tabla_ui(self):
        for i in self.tree.get_children(): self.tree.delete(i)
        for _, r in self.df.iterrows():
            self.tree.insert("", "end", values=[r.get(c, "-") for c in self.columnas_tabla])

    def guardar_proyecto(self):
        ruta = filedialog.asksaveasfilename(defaultextension=".xlsx")
        if ruta:
            self.df.to_excel(ruta, index=False)
            messagebox.showinfo("Guardado", "Proyecto exportado con éxito.")

if __name__ == "__main__":
    app = ProjectMasterUltra()
    app.root.mainloop()