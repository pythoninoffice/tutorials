from pcconfig import config
import pandas as pd
import pynecone as pc
import plotly.express as px
import plotly.graph_objects as go
from .mortgage_schedule import generate_mortgage_schedule


class State(pc.State):
    """The app state."""
    total_amount: float = 0
    down_payment: float = 0
    interest_rate: float = 0
    amort_period: int = 0
    monthly_amount: float = 0
    mortgage_amount: float = 0
    #mortgage_schedule: pd.DataFrame
    mortgage_schedule_fig: go.Figure = None
    
    # def ticker_update2(self):
    #     self.loading = True
    #     self.ticker2 = self.ticker
    #     self.loading = False
    
    #@pc.var
    def calc_monthly_amount(self) -> float:
        total_amount = float(self.total_amount)
        amort_period = int(self.amort_period)
        down_payment = float(self.down_payment)
        if float(self.interest_rate) != 0.0:
            interest_rate = float(self.interest_rate)/12
        
            print(self.total_amount)
            print(self.amort_period)
            print(self.interest_rate)
            if interest_rate > 0:
                annuity_factor = (1- (1+interest_rate) ** (-amort_period * 12)) / (interest_rate)
                if annuity_factor != 0:
                    self.monthly_amount = (total_amount-down_payment)/annuity_factor
                    #return self.monthly_amount
            else:
                #return 100
                pass
        else:
            #return 999
            pass
    #@pc.var
    def get_data(self)-> go.Figure:
        mortgage_schedule = generate_mortgage_schedule(self.total_amount, self.down_payment, self.interest_rate, self.amort_period)
        mortgage_schedule_df = pd.DataFrame(mortgage_schedule)
        #print(mortgage_schedule_df.shape)
        if mortgage_schedule_df.shape != (0,0):
            
            print(mortgage_schedule_df)
            fig = go.Figure(data = [
                go.Bar(name='Principal', x=mortgage_schedule_df['Month'], y=mortgage_schedule_df['Principal Payment']),
                go.Bar(name='Interest', x=mortgage_schedule_df['Month'], y=mortgage_schedule_df['Interest Payment']),
                ])
            
            self.mortgage_schedule_fig = fig
            self.mortgage_amount = float(self.total_amount) - float(self.down_payment)
            self.monthly_amount = round(mortgage_schedule_df['Total Payment'][0],2)
        
        
    
    # @pc.var
    # def line_chart(self) -> go.Figure:
    #     return px.line(self.df1, x=self.df1.index, y = 'Close', title='chart')

def get_input_field(icon: str, placeholder: str, _type:str, on_change = None):
    return pc.container(
        pc.hstack(
            pc.icon(
                tag=icon,
                color='white',
                fontSize='11px',
            ),
            pc.input(
                placeholder=placeholder,
                border = '0px',
                focus_border_color='None',
                color='white',
                fontWeight='semibold',
                fontSize='11px',
                type=_type,
                on_change = on_change
            ),
        ),
        borderBottom="0.1px solid grey",
        width="200px",
        height='45px',
    )
    
    


def index() -> pc.Component:
    calculator_container = pc.hstack(
        pc.box(width = "10%"),
        pc.box(pc.container(
            pc.vstack(
                pc.container(height="65px"),
                pc.container(
                    pc.text(
                        "Mortgage Calculator",
                        fontSize="28px",
                        color="white",
                        fontWeight="bold",
                        letterSpacing="3px"
                    ),
                    width="400px",
                    center_content=True,
                ),
                
                pc.container(
                    pc.text(
                        "A pynecone app example",
                        fontSize="12px",
                        color="white",
                        fontWeight="#eeeeee",
                        letterSpacing="0.25px"
                    ),
                    width="400px",
                    center_content=True,
                ),
                pc.container(height="50px"),
                get_input_field('star', 'Total Amount', '', on_change=State.set_total_amount),
                get_input_field('star', 'Down Payment', '', on_change=State.set_down_payment),
                get_input_field('star', 'Interest Rate', '', on_change=State.set_interest_rate),
                get_input_field('star', 'Amortization Period', '', on_change=State.set_amort_period),        
               
                pc.container(height='55px'),
                pc.container(
                    pc.button(
                        pc.text(
                            'Calculate',
                            color='white',
                            fontSize='11px',
                            weight='bold',
                        ),
                        on_click=State.get_data,
                        width='200px',
                        height='45px',
                        color_scheme='blue',
                    ),
                    center_content=True
                ),
                ),
            width = "400px",
            height = "75vh",
            center_content=True,
            bg = "#1D2330",
            borderRadius = '15px',
            boxShadow="41px -41px 82px #0d0f15, -41px 41px 82px #2d374b"), width="30%"),
        
        pc.box(
            pc.text('Total mortgage amount: $' + State.mortgage_amount),
            pc.text('Amortization Period: ' + State.amort_period * 12 + ' months.'),
            pc.text('Your monthly payment is: $' + State.monthly_amount),
            pc.container(
                pc.plotly(data=State.mortgage_schedule_fig, layout={'barmode':'stack', 'paper_bgcolor':'#1D2330', 'plot_bgcolor':'#1D2330', 
                                                                    'font':{'color':"white",
                                                                    'size':'20',},
                                                                    
                                                                    'width':'800'}, width='100%'),
                center_content = True,
                justifyContent= "center",
            ),
            color='white',
            width="50%"),
        pc.box(width="10%"),
        width = "100%"
    )

    
    
    _main = pc.container(
        calculator_container,
        center_content = True,
        justifyContent= "center",
        maxWidth="auto",
        height = "100vh",
        bg="#1D2330",
    )
    
    return _main


# Add state and page to the app.
app = pc.App(state=State)
app.add_page(index)
app.compile()
