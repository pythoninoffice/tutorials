"""Welcome to Pynecone! This file outlines the steps to create a basic app."""
from pcconfig import config

import pynecone as pc
import plotly.express as px
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from .navbar import navbar

filename = f"{config.app_name}/{config.app_name}.py"

text_style = {
    "color": "green",
    "font_family": "Comic Sans MS",
    "font_size": "1.2em",
    "font_weight": "bold",
    "box_shadow": "rgba(1, 1, 1, 0.8) 5px 5px, rgba(1, 1, 1, 0.4) 10px 10px",
}

class State(pc.State):
    """The app state."""
    ticker: str = "Stock Symbol"
    ticker2: str = ""
    username: str = "Jay"
    logged_in: bool = True
    loading: bool = False
    
    def ticker_update2(self):
        self.loading = True
        self.ticker2 = self.ticker
        self.loading = False
    
    @pc.var
    def df1(self) ->pd.DataFrame:
        stock = yf.Ticker(self.ticker2)
        df = stock.history(period='1y')
        return df
    
    @pc.var
    def line_chart(self) -> go.Figure:
        return px.line(self.df1, x=self.df1.index, y = 'Close', title='chart')
    
def index() -> pc.Component:
    return pc.center(
        pc.vstack(
            navbar(State),
            pc.text('Enter a stock symbol', 
                    background_image="linear-gradient(271.68deg, #EE756A 0.75%, #756AEE 88.52%)",
                    background_clip="text",
                    font_weight="bold",
                    font_size="2em",),
            pc.text(State.ticker2, font_family="Silkscreen",),
            pc.input(on_change=State.set_ticker),
            pc.button('update!', on_click=State.ticker_update2,style=text_style),
            pc.plotly(data=State.line_chart, layout={"width": "800", "height": "400"}),
            pc.box(
                pc.data_table(data=State.df1,
                          pagination=True,
                search=True,
                sort=True,
                resizable=True),
                width="1000px",
                height="500px",
                font_size="0.3em"      
            ),
            spacing="1.5em",
            font_size="2em",
        ),
        padding_top="10%",
    )

# Add state and page to the app.
app = pc.App(state=State,
             stylesheets=["https://fonts.googleapis.com/css2?family=Silkscreen&display=swap",],)        
app.add_page(index)
app.compile()