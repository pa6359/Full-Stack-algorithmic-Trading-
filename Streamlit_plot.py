import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import yfinance as yf
st.set_page_config(layout='wide')
class StreamlitStaticalDataPloter:
    
    def __init__(self, data):
        self.df = data


    def process_profit_and_loss(self):
        
        self.df['p/l'] = self.df["p/l"] - self.df["Expenses"]    
    
    def calculate_additional_columns(self):
        self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce').dt.strftime('%m-%d-%Y')
        
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        # Add the 'Day' column
        self.df['Day'] = self.df['Date'].dt.day_name()
        
        # Add the 'Month' column in 'Year-Month' format
        self.df['Month'] = self.df['Date'].dt.strftime('%Y-%m')

        # Add the 'Year' column in 'Year' format
        self.df['Year'] = self.df['Date'].dt.strftime('%Y')

        # Custom function to generate labels
        def week_label(date):
            week_number = (date.day - 1) // 7 + 1
            yearmonth = date.strftime('%Y-%m')
            return f"{yearmonth}-{week_number}W"

        # Apply the custom function to create the 'Week' column
        self.df['Week'] = self.df['Date'].apply(week_label)
        self.df = self.df.sort_values(by='Date').reset_index(drop=True) 
        self.df['Date'] = pd.to_datetime(self.df['Date'])
  
        
    def select_and_filter_columns(self):
        column_selectors = st.sidebar.multiselect("Select columns to filter:", list(self.df.columns))
        selected_values = {}

        for column_selector in column_selectors:
            unique_values = self.df[column_selector].unique()
            value_selector = st.sidebar.multiselect(f"Select values for {column_selector}:", ["Select All"] + list(unique_values))

            if "Select All" in value_selector:
                selected_values[column_selector] = list(unique_values)
            else:
                selected_values[column_selector] = value_selector

        for column, values in selected_values.items():
            self.df = self.df[self.df[column].isin(values)].reset_index(drop=True)

    
    def display_summary_table(self):
        # Assuming you have calculated the total profit/loss as 'total_p/l' earlie
   

        
        if self.df is not None:

            top5_max_pnl = self.df.nlargest(5, 'p/l')
            bottom5_min_pnl = self.df.nsmallest(5, 'p/l')
            
            st.subheader('Five Maximum Profits', divider='rainbow')
            st.table(top5_max_pnl)
            st.subheader('Five Maximum Loss', divider='rainbow')
            st.table(bottom5_min_pnl)

           

            
            grouping_option = st.sidebar.radio("Group summary tables by:", ['Strategy', 'Symbol'])

            if grouping_option:
                group_column = 'Strategy'
                symbol_column = 'Symbol'

                if grouping_option == 'Symbol':
                    group_column = 'Symbol'

                group_values = self.df[group_column].unique()

                summary_data = {
                    'Strategy': [],
                    'Symbol Type': [],
                    'Total P/L': [],
                    'Max No. of  Trade Daywise':[],
                    'Min No. of  Trade Daywise':[],
                    'Avg No. of  Trade Daywise':[],
                    'Max Fund utilization':[],
                    'Avg P/L per Trade': [],
                    'No. of Trades': [],
                    'Win Rate': [],
                    'No. of win Trades':[],
                    'Avg p/l Win Trade': [],
                    'No. of loss Trades':[],
                    'Avg p/l Lose Trade': [],
                    'Max Win Trade': [],
                    'Min Lose Trade': [],
                    'Profit Factor': [],
                    'Risk-Reward Ratio': [],
                    'Max-Drawdown': [],
                    'Drawdown %': [],
                    
                    

                }

                for group_value in group_values:
                    filtered_ = self.df[self.df[group_column] == group_value]
                    symbol_values = filtered_[symbol_column].unique()

                    for symbol_value in symbol_values:
                        filtered_df = filtered_[filtered_[symbol_column] == symbol_value]



                        # Identify time frame with the most occurrences of positive P/L for each date
                        consistent_profit_timeframe = filtered_df[filtered_df['p/l'] > 0].groupby('Date')['EnTime'].value_counts().groupby('Date').idxmax()
                        st.subheader('Consistent Profit', divider='rainbow')
                        result_df = pd.DataFrame(consistent_profit_timeframe.values.tolist(), columns=['Date', 'Timeframe'], index=consistent_profit_timeframe.index)
                        st.table(result_df)


                        total_profit_loss = filtered_df['p/l'].sum()
                        average_pl = total_profit_loss / len(filtered_df)
                        win_rate = (len(filtered_df[filtered_df['p/l'] > 0]) / len(filtered_df)) * 100
                        win_trade_df = filtered_df[filtered_df['p/l'] > 0]['p/l']
                        lose_trade_df = filtered_df[filtered_df['p/l'] < 0]['p/l']
                        max_fund_utilization = (filtered_df['BPrice'] * filtered_df['Qty']).max()
                        average_pl_winning_trade = win_trade_df.mean()
                        average_pl_losing_trade = lose_trade_df.mean()
                        largest_winning_trade = filtered_df[filtered_df['p/l'] > 0]['p/l'].max()
                        largest_losing_trade = filtered_df[filtered_df['p/l'] < 0]['p/l'].min()
                        total_profits = filtered_df[filtered_df['p/l'] > 0]['p/l'].sum()
                        total_losses = filtered_df[filtered_df['p/l'] < 0]['p/l'].sum()
                        profit_factor = total_profits / abs(total_losses) if total_losses != 0 else None
                        risk_reward_ratio = average_pl_winning_trade / abs(average_pl_losing_trade) if average_pl_losing_trade != 0 else None
                        cumulative_pl = filtered_df['p/l'].cumsum()
                        max_drawdown = (cumulative_pl - cumulative_pl.expanding().max()).min()
                        drawdown_percentage = ((max_drawdown * 100) / cumulative_pl.max())

                        if group_value == symbol_value:
                            summary_data['Strategy'].append(group_value)
                            summary_data['Symbol Type'].append(None)
                        else:
                            summary_data['Strategy'].append(group_value)
                            summary_data['Symbol Type'].append(symbol_value)

                        summary_data['Total P/L'].append(int(total_profit_loss))
                        summary_data['Max No. of  Trade Daywise'].append(int(filtered_df['total_trade_per_day'].max()))
                        summary_data['Min No. of  Trade Daywise'].append(int(filtered_df['total_trade_per_day'].min()))
                        summary_data['Avg No. of  Trade Daywise'].append(int(filtered_df['total_trade_per_day'].mean()))                       
                        summary_data['Max Fund utilization'].append(int(max_fund_utilization))
                        summary_data['Avg P/L per Trade'].append(int(average_pl))
                        summary_data['No. of Trades'].append(len(filtered_df))
                        summary_data['Win Rate'].append(win_rate)
                        summary_data['No. of win Trades'].append(len(win_trade_df))
                        summary_data['Avg p/l Win Trade'].append(int(average_pl_winning_trade))
                        summary_data['No. of loss Trades'].append(len(lose_trade_df))
                        summary_data['Avg p/l Lose Trade'].append(int(average_pl_losing_trade))
                        summary_data['Max Win Trade'].append(int(largest_winning_trade))
                        summary_data['Min Lose Trade'].append(int(largest_losing_trade))
                        summary_data['Profit Factor'].append(profit_factor)
                        summary_data['Risk-Reward Ratio'].append(risk_reward_ratio)
                        summary_data['Max-Drawdown'].append(int(max_drawdown))
                        summary_data['Drawdown %'].append(drawdown_percentage)

                summary_table = pd.DataFrame(summary_data)
                
                summary_table_transposed = summary_table.T
              
                # Styling the table
                styled_table = summary_table_transposed.style \
                    .applymap(lambda x: 'color: green' if (isinstance(x, (int, float)) and x >= 0) else ('color: red' if (isinstance(x, (int, float)) and x < 0) else '')) \
                    .set_table_styles([
                        {'selector': 'tr:first-child', 'props': [('background-color', 'turquoise')]},  # Background color for the first row
                        {'selector': 'th', 'props': [('background-color', 'turquoise')]},  # Background color for the header cells
                        {'selector': 'tr td:first-child', 'props': [('background-color', 'turquoise')]},  # Background color for the first column
                        {'selector': '', 'props': [('border', '8px solid #dddddd')]}  # Add border to all cells
                    ])
                st.subheader('DATA ANALYSIS AND STATS', divider='rainbow')
                st.table(styled_table)


            else:
                st.error("Invalid option selected")


    def display_cumulative_line_graph(self):
    
        strategy_df =  self.df.groupby(['Strategy','Date']).agg({'p/l': 'sum'}).reset_index()
     
        max_date = strategy_df['Date'].max()
        # Calculate overall p/l
        overall_pl = strategy_df.groupby(['Strategy'])['p/l'].sum().reset_index()
        overall_pl = overall_pl.rename(columns={'p/l': 'overall p/l'})

        # Filter for the max date and reset the index
        max_date_data = strategy_df[strategy_df['Date'] == max_date].copy().reset_index(drop=True)
        max_date_data = max_date_data.rename(columns={'p/l': 'today p/l'})

        # Merge the overall p/l and today p/l DataFrames
        result_df = overall_pl.merge(max_date_data, on='Strategy', how='left')
        result_df['today p/l'] = result_df['today p/l'].fillna(0)
        result_df['Date'] = result_df['Date'].fillna(max_date)
        result_df['Margin'] = 0
        unique_strategies = result_df['Strategy'].unique()

        # Dictionary to store strategy-margin mapping
        strategy_margin_mapping = {}

        # Iterate through unique_strategies
        for strategy in unique_strategies:
            # Get margin input from the user using Streamlit's number_input
            margin_input = st.number_input(
                f"Enter the margin value for strategy '{strategy}':",
                min_value=10000,
                max_value=2000000,
                value=100000,
                step=20000
            )

            # Store the margin input in the dictionary
            strategy_margin_mapping[strategy] = margin_input

        # Create the 'Margin' column using the 'Strategy' column and the mapping
        result_df['Margin'] = result_df['Strategy'].map(strategy_margin_mapping)

        if int(result_df['Margin']) != 0:
            result_df['ROI%'] = (result_df['overall p/l']/ result_df['Margin'])*100
        else:
            result_df['ROI%'] = 0

# Assuming you have calculated the total profit/loss as 'total_p/l' and total expenses as 'total_expenses' earlier
        total_p_l = result_df['overall p/l'].sum()
        total_expenses = self.df['Expenses'].sum()
        Today_p_l = result_df['today p/l'].sum()
        # Define a function to determine the text color based on the value
        def get_text_color(value):
            if value < 0:
                return 'red'
            else:
                return 'green'

        # Create a two-column layout
        left_column, right_column ,center_column= st.columns(3)

        # Display "Total Expenses" on the left side with conditional styling
        with right_column:
            text_color = get_text_color(total_expenses)
            st.write(f"<div style='background-color: yellow; padding: 10px; color: {text_color};'><b>Total Expenses:</b> {int(total_expenses)}</div>", unsafe_allow_html=True)
            
        with left_column:
            text_color = get_text_color(Today_p_l)
            st.write(f"<div style='background-color: pink; padding: 10px; color: {text_color};'><b><b>Today P/L:</b> {int(Today_p_l)}</div>", unsafe_allow_html=True)
            
        # Display "Total Profit/Loss" on the right side with conditional styling
        with center_column:
            text_color = get_text_color(total_p_l)
            st.write(f"<div style='background-color: lightblue; padding: 10px; color: {text_color};'><b>Overall P/L:</b> {int(total_p_l)}</div>", unsafe_allow_html=True)
     
# Create a figure for the combined cumulative graph with 5 subplots
        fig_cumulative = make_subplots(rows=6, cols=1, shared_yaxes=True, horizontal_spacing=0.05, vertical_spacing = 0.1)
        
        # Define the width of each bar and the number of strategies
        bar_width = 0.35
        num_strategies = len(strategy_df)

        # Create an array for x-coordinates for each set of bars
        x = np.arange(num_strategies)

           
        # Add both "Today P/L" and "Overall P/L" bars with the same strategy name
        fig_cumulative.add_trace(go.Bar(
            x=x - bar_width/2,  # Adjust x-coordinates to place bars side by side
            y=result_df['today p/l'],
            text=result_df['today p/l'].apply(lambda x: f'<b>{x:.2f}</b>'),
            width=bar_width,
            marker_color=['lightblue' if val > 0 else 'orange' for val in result_df['today p/l']],
            name='Today P/L',  # Use a common name for both bars
            hovertemplate='<b>%{text}</b>'
        ), row=1, col=1)

        fig_cumulative.add_trace(go.Bar(
            x=x + bar_width/2,  # Adjust x-coordinates to place bars side by side
            y=result_df['overall p/l'],
            text=result_df['overall p/l'].apply(lambda x: f'<b>{x:.2f}</b>'),
            width=bar_width,
            marker_color=['blue' if val > 0 else 'red' for val in result_df['overall p/l']],
            name='Overall P/L',  # Use the same common name for both bars
            hovertemplate='<b>%{text}</b>'
        ), row=1, col=1)
        
        # Add a third bar for ROI%
        fig_cumulative.add_trace(go.Bar(
            x=x+2*bar_width/2+0.1,
            y=result_df['ROI%'],
            text=result_df['ROI%'].apply(lambda x: f'<b>{x:.2f}</b>'),
            width=0.2,
            marker_color=['green' if val > 0 else 'purple' for val in result_df['ROI%']],
            name='ROI%',
            hovertemplate='<b>%{text}</b>'
        ))


        # Set the x-axis tick text to the strategy names for the first subplot
        fig_cumulative.update_xaxes(
            ticktext=result_df['Strategy'],  # Set strategy names as x-axis labels
            tickfont=dict(size=20, color='black'),
            tickvals=x,  # Position of the labels on the x-axis
            row=1, col=1
        )
        
        
      
        weekly_df = self.df.groupby('Week')['p/l'].sum().reset_index()
      
        monthly_df = self.df.groupby('Month')['p/l'].sum().reset_index()

        Yearly_df = self.df.groupby('Year')['p/l'].sum().reset_index()
    
        last_index = len(weekly_df) - 1
        first_index_to_show = last_index - 6


            # Define the data for month-wise, week-wise, and date-wise cumulative P/L line graphs
        graph_data = [
            (weekly_df, 'Cumulative P/L (Week-wise)'),
            (monthly_df, 'Cumulative P/L (Month-wise)'),
            (Yearly_df, 'Cumulative P/L (Year-wise)'),
            
        ]

        # Add cumulative P/L line graphs and annotations
        for i, (data, graph_name) in enumerate(graph_data):
            
            
            
            # Add both "Today P/L" and "Overall P/L" bars with the same strategy name
            fig_cumulative.add_trace(go.Bar(
                x=data.index, # Adjust x-coordinates to place bars side by side
                y=data['p/l'],
                text=data['p/l'].apply(lambda x: f'<b>{x:.2f}</b>'),
                width=bar_width,
                marker_color=['lightblue' if val > 0 else 'orange' for val in data['p/l']],
                name=f'{graph_name} P/L',  # Use a common name for both bars
                hovertemplate='<b>%{text}</b>' 
            ), row=(4+i) , col=1)
                
            
            fig_cumulative.add_trace(go.Scatter(
                x=data.index,
                y=data['p/l'].cumsum(),
                mode='lines+markers',  # Add markers to the line
                name=graph_name,
                text=graph_name,  # Text to display on hover (cumulative P/L values)
                line=dict(width=5),
            ), row=(4+i) , col=1)

            # Add annotations for the cumulative line graph
            for j, value in enumerate(data['p/l'].cumsum()):
                fig_cumulative.add_annotation(
                    x=data.index[j],
                    y=value,
                    text=f'<b>{value:.0f}</b>',  # Format the text as needed
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowcolor='black',
                    arrowwidth=1,
                    bgcolor='yellow',
                    bordercolor='black',
                    borderwidth=1,
                    font=dict(color='black', size=15, family='Arial'),
                    row=(4+i) , 
                    col=1
                )
         # Set x-axis labels and title for row 1, column 2 (Week)
        fig_cumulative.update_xaxes(
            showline=True, 
            showgrid=False,
            rangeslider_visible=True,
            rangeslider_thickness=.005,
            range=[first_index_to_show, last_index],
            ticktext=weekly_df['Week'],  # Set Week as x-axis labels
            tickfont=dict(size=15, color='black'),
            tickvals=weekly_df.index,  # Position of the labels on the x-axis
            title_text="<b>Equity curve Weekly</b>",  # Set the x-axis title
            row=4, col=1
        )


        # Set x-axis labels and title for row 2, column 2 (Month)
        fig_cumulative.update_xaxes(
            showline=True, 
            showgrid=False,
            rangeslider_thickness=.005,
            rangeslider_visible=True,
            ticktext=monthly_df['Month'],  # Set Month as x-axis labels
            tickfont=dict(size=18, color='black'),
            tickvals=monthly_df.index,  # Position of the labels on the x-axis
            title_text="<b>Equity curve Monthly</b>",
            # Set the x-axis title
            row=5, col=1
        )
         
        # Set x-axis labels and title for row 2, column 2 (Month)
        fig_cumulative.update_xaxes(
            showline=True, 
            showgrid=False,
            rangeslider_thickness=.005,
            rangeslider_visible=True,
            ticktext=Yearly_df['Year'],  # Set Month as x-axis labels
            tickfont=dict(size=18, color='black'),
            tickvals=Yearly_df.index,  # Position of the labels on the x-axis
            title_text="<b>Equity curve Yearly</b>",
            # Set the x-axis title
            row=6, col=1
        ) 
#  daily cummulative graph    
        daily_cumulative = self.df.groupby('Date')['p/l'].sum().reset_index()
        # Create a Date-wise cumulative P/L line trace
        fig_cumulative.add_trace(go.Scatter(
            x=daily_cumulative.index,
            y=daily_cumulative['p/l'].cumsum(),
            mode='lines',  # Add markers to the line
            name='Cumulative P/L (Date-wise)',
            line=dict(color='blue', width=3),
            text=daily_cumulative['Date'],  # Text to display on hover (cumulative P/L values)
        ), row=2, col=1)
        
        # Add annotation for the last point
        last_index = daily_cumulative.index[-1]
        last_pl = daily_cumulative['p/l'].cumsum().iloc[-1]

        fig_cumulative.add_annotation(
            x=last_index,
            y=last_pl,
            text=f'<b>Cummulative P/l:({last_pl:.0f}), Today P/l:({Today_p_l:.0f})</b>',
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowcolor='black',
            arrowwidth=1,
            bgcolor='yellow',
            bordercolor='black',
            borderwidth=1,
            ax=0,
            ay=-40,
            row = 2,
            col = 1
        )
        
        fig_cumulative.update_xaxes(
            title_text="<b>Equity curve daily</b>",
            showline=True, 
            showgrid=True,
            ticktext=daily_cumulative['Date'],
            tickfont=dict(size=18, color='black'),
            rangeslider_thickness=.01,
            rangeslider_visible=True, 
            row=2, col=1
        )
        
# Sort the drawdown values in descending order and select the top 5
        
        daily_cumulative['Peak'] = daily_cumulative['p/l'].cummax()
        daily_cumulative['Drawdown'] = (daily_cumulative['p/l'] - daily_cumulative['Peak'])   
        top_drawdowns = daily_cumulative.sort_values(by='Drawdown', ascending=True).head(5)
        st.subheader('Five Maximum Drawdown', divider='rainbow')
        st.table(top_drawdowns)


        fig_cumulative.add_trace(
            go.Scatter(x=daily_cumulative.index, y=daily_cumulative['Drawdown'], mode='lines', name='Drawdown', text=daily_cumulative['Date'], line=dict(color='red', width=3)), row=3, col=1)

        # Highlight the top 5 drawdowns
        fig_cumulative.add_trace(
            go.Scatter(x=top_drawdowns.index, y=top_drawdowns['Drawdown'], mode='markers', name='Top 5 Drawdowns', marker=dict(color='blue', size=8)), row=3, col=1)

        fig_cumulative.update_xaxes(
            title_text="<b>DRAWDOWN CHART</b>",
            showline=True, 
            showgrid=True,
            ticktext=daily_cumulative['Date'],
            tickfont=dict(size=18, color='black'),
            rangeslider_thickness=.01,
            rangeslider_visible=True, 
            row=3, col=1
        )

# Add P/L bars trace with custom colors
        bar_colors = ['turquoise' if val > 0 else 'orange' for val in daily_cumulative['p/l']]
        bar_colors[daily_cumulative['p/l'].idxmax()] = 'blue'  # Set maximum P/L bar to blue
        bar_colors[daily_cumulative['p/l'].idxmin()] = 'red'  # Set minimum P/L bar to red

        fig_cumulative.add_trace(go.Bar(x=daily_cumulative.index, y=daily_cumulative['p/l'], marker_color=bar_colors, name = 'daywise p/l'), row=2, col=1)
        
        # Customize layout
        # fig_cumulative.update_yaxes(title_text='<b>Profit/Loss</b>', showline=True, showgrid=True ,tickfont=dict(size=18, color='black'))
        fig_cumulative.update_layout(height=2500, width=925 ,barmode='relative',legend=dict(x=0, y=1.1, traceorder='normal', orientation='h', itemclick='toggleothers'),font=dict(size=17, color='black'))

        # Display the chart
        st.plotly_chart(fig_cumulative, use_container_width=True)


        
    def download_financial_data(self, symbol, df):
       
        symbols_mapping = {
            'NIFTY': '^NSEI',        # Nifty 50
            'BANKNIFTY': '^NSEBANK',  # Bank Nifty
            'FINNIFTY': '^NSEBANK'    # Assuming FINNIFTY corresponds to Bank Nifty, you may need to adjust this
        }

        if symbol in symbols_mapping:
            # Get the corresponding symbol from the mapping
            symbol = symbols_mapping[symbol]

            # Extract min and max dates from the DataFrame
            min_date = df['Date'].min()
            max_date = df['Date'].max()

            # Convert min and max dates to string format
            start_date = pd.to_datetime(min_date).strftime('%Y-%m-%d')
            end_date = pd.to_datetime(max_date).strftime('%Y-%m-%d')

            # Download historical data
            data = yf.download(symbol, start=start_date, end=end_date, interval='1d')

            return data
        else:
            st.error(f"Symbol '{symbol}' not recognized. Please provide a valid symbol.")
            return None
    
    def tabs(self):
            # Get the unique symbols
        symbols = self.df["Symbol"].unique()
        st.header('Symbol wise charts compare with their underline Asset', divider='rainbow')
        # Create tabs for each unique symbol
        for symbol in symbols:
            
            st.subheader(f'{symbol}')
            
            symboldf = self.df[self.df["Symbol"] == symbol]
            
            # Daily cumulative graph
            daily_cumulative = symboldf.groupby('Date')['p/l'].sum().reset_index()
            
            underlinedata = self.download_financial_data(symbol, daily_cumulative)

            # Create a Date-wise cumulative P/L line trace
            fig_Symbol = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Adding traces for underlying and cumulative data
            fig_Symbol.add_trace(go.Scatter(x=underlinedata.index, y=underlinedata['Close'], mode='lines', name='Underlying',
                                     line=dict(color='green')),secondary_y=True)
            
            fig_Symbol.add_trace(go.Scatter(
                x=daily_cumulative['Date'],
                y=daily_cumulative['p/l'].cumsum(),
                mode='lines',
                name= f'Cumulative {symbol}',
                line=dict(color='blue', width=3),
                text=daily_cumulative['Date']
            ))
            
            # Add annotation for the last point
            last_index = daily_cumulative['Date'].max()
            last_pl = daily_cumulative['p/l'].cumsum().iloc[-1]

            
            fig_Symbol.add_annotation(
                x=last_index,
                y=last_pl,
                text=f'<b>Cummulative P/l:({last_pl:.0f}))</b>',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowcolor='black',
                arrowwidth=1,
                bgcolor='pink',
                bordercolor='black',
                borderwidth=1,
                ax=0,
                ay=-40,
            )

            fig_Symbol.update_layout(
                xaxis=dict(title='Date'),
                yaxis1=dict(title='Cumulative', titlefont=dict(color='blue'), tickfont=dict(color='blue')),
                yaxis2=dict(title='Underlying', titlefont=dict(color='lightgreen'), tickfont=dict(color='lightgreen'), overlaying='y',
                            side='right'),            
                legend=dict(x=.3, y=1.1, traceorder='normal', orientation='h', itemclick='toggleothers')
            )
            
            # Display the Plotly figure
            st.plotly_chart(fig_Symbol,  use_container_width=True)
    
    def weekday(self):

        on = st.sidebar.toggle('open Weekday charts')
        if on:
            weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            st.header('Weekday Charts', divider='rainbow')

            # Create 2 columns and 3 rows for Monday to Friday
            fig_weekday = make_subplots(rows=3, cols=2, shared_yaxes=False, vertical_spacing=0.2, subplot_titles=weekdays)

            for i, weekday in enumerate(weekdays, start=0):
                weekdaydf = self.df[self.df["Day"] == weekday]

                # Daily cumulative graph
                daily_cumulative = weekdaydf.groupby('Date')['p/l'].sum().reset_index()

                fig_weekday.add_trace(go.Scatter(
                    x=daily_cumulative['Date'],
                    y=daily_cumulative['p/l'].cumsum(),
                    mode='lines',
                    name=f'{weekday} cumulative',
                    line=dict(color='red', width=5),
                    text=daily_cumulative['Date']
                ), row=(i // 2 ) + 1, col=(i % 2) + 1)

                # Add annotation for the last point
                last_index = daily_cumulative['Date'].max()
                last_pl = daily_cumulative['p/l'].cumsum().iloc[-1]

                
                fig_weekday.add_annotation(
                    x=last_index,
                    y=last_pl,
                    text=f'<b>Cummulative P/l:({last_pl:.0f}))</b>',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowcolor='black',
                    arrowwidth=1,
                    bgcolor='turquoise',
                    bordercolor='black',
                    borderwidth=1,
                    ax=0,
                    ay=-40,
                    row=(i // 2 ) + 1,
                    col=(i % 2) + 1
                )

            # Update layout for better appearance
            fig_weekday.update_layout(height=1200, showlegend=False)

            # Display the Plotly figure (outside the loop)
            st.plotly_chart(fig_weekday, use_container_width=True)
    
    def style_data(self,style_df):
        
        styled_table = style_df.style \
            .applymap(lambda x: 'color: green' if (isinstance(x, (int, float)) and x >= 0) else ('color: red' if (isinstance(x, (int, float)) and x < 0) else '')) \
            .set_table_styles([
                {'selector': 'tr:last-child', 'props': [('background-color', 'turquoise')]},  # Background color for the last row
                {'selector': 'th', 'props': [('background-color', 'turquoise')]},  # Background color for the header cells
                {'selector': 'tr td:first-child', 'props': [('background-color', 'turquoise')]},  # Background color for the first column
                {'selector': 'tr td:last-child', 'props': [('background-color', 'turquoise')]},  # Background color for the last column
                {'selector': '', 'props': [('border', '8px solid #dddddd')]}  # Add border to all cells
            ])
        return styled_table
    
    def trade_table(self):
    
        # Calculate the length of unique dates and total P/L for each combination of 'total_trade_per_day' and 'Day'
        grouped_data = self.df.groupby(['total_trade_per_day', 'Day'])
        unique_dates_lengths = grouped_data['Date'].unique().apply(len).reset_index(name='UniqueDatesLength')
        total_pl_per_group = grouped_data['p/l'].sum().reset_index(name='TotalPL')

        # Merge the two DataFrames on the common columns
        merged_data = pd.merge(unique_dates_lengths, total_pl_per_group, on=['total_trade_per_day', 'Day'])
        # Create a crosstab using 'total_trade_per_day' and 'Day'
        cross_tab = pd.crosstab(index=self.df['total_trade_per_day'], columns=self.df['Day'])
        cross_tab2 = pd.crosstab(index=self.df['total_trade_per_day'], columns=self.df['Day'])

        # Iterate over each cell in the crosstab and fill in the corresponding values
        for index, value in merged_data.iterrows():
            total_trade_per_day = value['total_trade_per_day']
            day = value['Day']
            unique_dates_length = value['UniqueDatesLength']
            total_pl = value['TotalPL']
            
            # Use .loc to put the calculated value in the crosstab
            cross_tab.loc[total_trade_per_day, day] = unique_dates_length
            cross_tab2.loc[total_trade_per_day, day] = int(total_pl)

        # Add a column for total days trade-wise (sum along rows)
        cross_tab['TotalDaysTradeWise'] = cross_tab.sum(axis=1)
        cross_tab2['Total_p/l_TradeWise'] = cross_tab2.sum(axis=1)
        # Add an index for total days day-wise (sum along columns)
        cross_tab.loc['TotalDaysDayWise'] = cross_tab.sum(axis=0)
        cross_tab2.loc['Total_p/l_DayWise'] = cross_tab2.sum(axis=0)
       
        cross_tab = self.style_data(cross_tab)
        cross_tab2 = self.style_data(cross_tab2)
        
        st.subheader('TRADEs-weekDAY Total Days Sheet', divider='rainbow')
        st.table(cross_tab)
        st.subheader('TRADEs-weekDAY Total P/L Sheet', divider='rainbow')
        st.table(cross_tab2)

    def tradesheet(self):
        # Assuming df is your DataFrame
        columns_to_drop = ['Month', 'Week', 'Strategy', 'Volume', 'Expenses', 'total_trade_per_day']

        # Drop the specified columns
        self.df = self.df.drop(columns=columns_to_drop)
        
        st.table(self.df)
               
    def runplot(self):
        self.process_profit_and_loss()
        self.calculate_additional_columns()
        self.select_and_filter_columns()
        self.display_cumulative_line_graph()
        self.display_summary_table()
        self.tabs()
        self.trade_table()
        self.weekday() 
        self.tradesheet()
        
# if __name__ == "__main__":
#     trader = StreamlitStaticalDataPloter()
#     trader.runplot()
