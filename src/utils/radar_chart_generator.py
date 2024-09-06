import textwrap
import time
import plotly.express as px
import plotly.graph_objects as go
from const import beautified_model_names, distinctive_colors


class RadarChartPlotter:
    _instance = None  # Singleton instance variable

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RadarChartPlotter, cls).__new__(cls)
            # Initialize the object here if needed
            cls._instance.color_map = cls._instance.get_color_map()
        return cls._instance

    @staticmethod
    def customwrap(s, width=20):
        # remove reasoning from the model name
        s = s.replace(" Knowledge", "") if s!="Knowledge Base" else s
        return "<br>".join(textwrap.wrap(s, width=width))

    @staticmethod
    def get_color_map():
        all_models = beautified_model_names.values()
        colors = distinctive_colors
        color_map = {model: colors[i % len(colors)] for i, model in enumerate(all_models)}
        return color_map

    def plot_radar_chart(self, df, inclusion_num_instances_threshold, included_models, file_path):
        range_val = [0, 90]
        tickvals = list(range(0, 91, 10))
        # remove rows where num_instances is less than threshold
        df = df[df['num_instances'] >= inclusion_num_instances_threshold]
        fig = go.Figure()
        for column in included_models:
            values = df[column].values.tolist() + [df[column].values[0]]
            indices = df.index.tolist() + [df.index[0]]
            indices = [self.customwrap(index, 50) for index in indices]
            model_color = self.color_map[beautified_model_names.get(column, column)]
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=indices,
                fill='none',
                name=column,
                line=dict(width=5, color=model_color),
                marker=dict(size=12, color=model_color),
                mode='lines+markers'
            ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=range_val,
                    tickvals=tickvals,
                    tickfont=dict(
                        size=30,
                        family='Helvetica'
                    ),
                    titlefont=dict(
                        size=30,
                        family='Helvetica'
                    )
                ),
                angularaxis=dict(
                    tickfont=dict(
                        size=30,
                        family='Helvetica'
                    )
                )
            ),
            legend=dict(
                x=0.5,
                y=1.15,
                xanchor='center',
                orientation='h',
                font=dict(
                    size=26,
                    family='Helvetica'
                ),
                bgcolor='#e5ecf6',
                bordercolor='Black',
                borderwidth=1
            ),
            margin=dict(l=0, r=0)
        )
        fig1 = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
        fig1.write_image(file_path, width=1500, height=1200, scale=2)
        time.sleep(3)
        fig.write_image(file_path, width=1500, height=1200, scale=2)
        # purposely saving the figure twice to avoid MathJax watermark bug