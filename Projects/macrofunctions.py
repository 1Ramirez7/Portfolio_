import parameters
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm





def update_excel_df(excel_file_path, df): # to update df macrodata excel file
    excel_df = pd.read_excel(excel_file_path)

    # Check if the number of rows in excel_df is less than df and deletes any excess rows
    if len(excel_df) < len(df):
        additional_rows = len(df) - len(excel_df)
        excel_df = pd.concat([excel_df, pd.DataFrame([{}]*additional_rows)], ignore_index=True)

    # Update relevant columns with the data from df
    for column in df.columns:
        if column in excel_df.columns:
            excel_df[column] = df[column]
    # Write the updated DataFrame to the Excel file
    excel_df.to_excel(excel_file_path, index=False)
# sample use
# update_excel_df("C://Users//eduar//OneDrive - BYU-Idaho//Desktop//Coding//Macro//macrodata.xlsx", df)

def update_parameters_and_rewrite(new_values):
    def format_parameters_dict(parameters_dict):
        formatted_dict = "{\n"
        for k, v in parameters_dict.items():
            formatted_dict += f"    '{k}': {v},\n"
        formatted_dict += "}"
        return formatted_dict

    # Update the parameters
    for key, value in new_values.items():
        if key in parameters.parameters:
            parameters.parameters[key] = value
        else:
            print(f"Key '{key}' not found in parameters.")

    # Rewrite the parameters.py file
    with open('parameters.py', 'w') as file:
        file.write("# parameters.py\n\n")
        file.write(f"fed_target_initial_inflation = {parameters.fed_target_initial_inflation}\n")
        file.write(f"simulation_years = {parameters.simulation_years}\n")
        file.write(f"num_periods = {parameters.num_periods}\n")
        file.write("\nparameters = ")
        file.write(format_parameters_dict(parameters.parameters))





# model CH13 1 def functions
# to recall
# import pandas as pd
# from macrofunctions import get_value, calculate_inflation, calculate_SRY, calculate_Rt


def calculate_inflation1(df, row_index, b, m, v, pi_bar):
    if row_index == 0:
        inflation = (1 - (b * m * v) / (1 + b * m * v)) * df.at[row_index, 'pi_bar'] + \
               v * df.at[row_index, 'a'] / (1 + b * m * v) + \
               (b * m * v) / (1 + b * m * v) * pi_bar + \
               (1 - (b * m * v) / (1 + b * m * v)) * df.at[row_index, 'o(%)']
    else:
        inflation =  ((1 - (b * m * v) / (1 + b * m * v)) * df.at[row_index-1, 'inflation'] +
                v * df.at[row_index, 'a'] / (1 + b * m * v) +
                (b * m * v) / (1 + b * m * v) * df.at[row_index, 'pi_bar'] +
                (1 - (b * m * v) / (1 + b * m * v)) * df.at[row_index, 'o(%)'])
    return inflation

def calculate_SRY(df, row_index, b, m, v, pi_bar):
    if row_index == 0:
        SRY =  df.at[row_index, 'a'] - b * m * (pi_bar - df.at[row_index, 'pi_bar'] + df.at[row_index, 'o(%)']) / (1 + b * m * v)
    else:
        SRY = df.at[row_index, 'a'] - b * m * (df.at[row_index-1, 'inflation'] - df.at[row_index, 'pi_bar'] + df.at[row_index, 'o(%)']) / (1 + b * m * v)
    return SRY

def calculate_Rt(df, row_index, m):
    Rt = m * (df.at[row_index, 'inflation'] - df.at[row_index, 'pi_bar']) + 2
    return Rt

# End of CH13 1 functions.


# AD-AS Simulation Ch13 - Should do model

def calculate_inflation_stabilization_inflation(df, row_index, fed_target_initial_inflation):
    if row_index == 0:
        inflation = fed_target_initial_inflation   
    else:
        inflation = (df.at[row_index-1, 'Inflation Stabilization inflation'] + df.at[row_index, 'o(%)'] + (df.at[row_index, 'v'] * df.at[row_index, 'b'] * df.at[row_index, 'm']) * fed_target_initial_inflation + \
            df.at[row_index, 'a(%)'] * df.at[row_index, 'v']) / (1 + df.at[row_index, 'v'] * df.at[row_index, 'b'] * df.at[row_index, 'm'])

    return inflation

def calculate_inflation_stabilization_sroutput(df, row_index, fed_target_initial_inflation):
    if row_index == 0:
        sroutput = 0.0 
    else:
        sroutput = df.at[row_index, 'a(%)'] - (df.at[row_index, 'b'] * df.at[row_index, 'm']) / \
            (1 + df.at[row_index, 'v'] * df.at[row_index, 'b'] * df.at[row_index, 'm']) * \
            (df.at[row_index, 'Inflation Stabilization inflation'] - fed_target_initial_inflation + df.at[row_index, 'o(%)'] + df.at[row_index, 'a(%)'] * df.at[row_index, 'v'])
    return sroutput

def calculate_output_stabilization(df, row_index, fed_target_initial_inflation):
    if row_index == 0:
        sroutput = 0.0  # edit first period
        inflation = fed_target_initial_inflation 
    else:
        sroutput = df.at[row_index, 'a(%)'] / (1 + df.at[row_index, 'b'] * df.at[row_index, 'n'])  # sroutput formula
        inflation = df.at[row_index - 1, 'Output Stabilization inflation'] + \
                    df.at[row_index, 'v'] * sroutput + df.at[row_index, 'o(%)']  # inflation formula. note this formula has the sroutput formula in it. 
    
    return sroutput, inflation

def calculate_taylor_rule_sroutput(df, row_index, fed_target_initial_inflation):
    if row_index == 0:
        sroutput = 0.0  
    else:
        sroutput = (df.at[row_index, 'b'] * df.at[row_index, 'm']) / (1 + df.at[row_index, 'b'] * df.at[row_index, 'n'] + df.at[row_index, 'v'] * df.at[row_index, 'b'] * df.at[row_index, 'm']) * \
            (df.at[row_index, 'a(%)'] / df.at[row_index, 'b'] * df.at[row_index, 'm'] - \
            df.at[row_index - 1, 'Taylor Rule inflation'] + fed_target_initial_inflation - df.at[row_index, 'o(%)']) 
    return sroutput

def calculate_taylor_rule_inflation(df, row_index, fed_target_initial_inflation):
    if row_index == 0:
        inflation = fed_target_initial_inflation 
    else:
        inflation = df.at[row_index - 1, 'Taylor Rule inflation'] + \
                    df.at[row_index, 'o(%)'] + df.at[row_index, 'v'] * df.at[row_index, 'Taylor Rule sroutput']
    return inflation



# end of chapter 13 Ad-As simulation


# 12CH 1 Short-run Simulation

# calculate_y_tilde function
def calculate_y_tilde(df, row_index):
    Y_tilde =  df.at[row_index, 'a(%)'] - df.at[row_index, 'b'] * (df.at[row_index, 'R(%)'] - df.at[row_index, 'r(%)'])
    return Y_tilde

# calculate_c_inflation function
def calculate_c_inflation(df, row_index):
    c_inflation = df.at[row_index, 'Y_tilde'] * df.at[row_index, 'v'] + df.at[row_index, 'o(%)']
    return c_inflation

# Dcalculate_inflation function
def calculate_inflation(df, row_index, fed_target_initial_inflation):
    if row_index == 0:
        return fed_target_initial_inflation
    else:
        inflation = df.at[row_index-1, 'inflation'] + df.at[row_index, 'c_inflation']
    return inflation

# end of 12 CH
    

# graphs


def plot_1_vs_period(dataframe, y_variable):
    if y_variable in dataframe.columns:
        plt.scatter(dataframe['Period'], dataframe[y_variable])
        plt.plot(dataframe['Period'], dataframe[y_variable], linestyle='-', color='blue')
        plt.xlabel('Period')
        plt.ylabel(y_variable)
        plt.title(f'Parameter {y_variable}')
        plt.xticks(range(0, dataframe['Period'].max() + 1))
        plt.grid(True)
        plt.show()
    else:
        print(f"Column '{y_variable}' not found in DataFrame.")
# sample usage 
# plot_1_vs_period(df, 'a(%)')

# mulitple plots with one variable each
def plot_m_vs_period(dataframe, y_variables, x_tick_interval=None, custom_x_ticks=None):
    num_vars = len(y_variables)
    num_rows = (num_vars + 3) // 4  # Calculate number of rows needed
    num_cols = min(num_vars, 4)     # Maximum of 4 columns

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
    axs = axs.flatten()  # Flatten in case of a single row

    for i, var in enumerate(y_variables):
        if var in dataframe.columns:
            # axs[i].scatter(dataframe['Period'], dataframe[var]) # comment in to add dots in line
            axs[i].plot(dataframe['Period'], dataframe[var], linestyle='-', color='blue')
            axs[i].set_xlabel('Period')
            axs[i].set_ylabel(var)
            axs[i].set_title(f'Parameter: {var}')
            
            # Set custom x-axis ticks if provided, otherwise use interval
            if custom_x_ticks is not None:
                axs[i].set_xticks(custom_x_ticks)
            elif x_tick_interval is not None:
                axs[i].set_xticks(range(0, dataframe['Period'].max() + 1, x_tick_interval))

            axs[i].grid(True)
        else:
            print(f"Parameter '{var}' not found in DataFrame.")

    # Hide any unused subplots
    for j in range(i + 1, num_rows * num_cols):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()

# Example usage:
# plot_m_vs_period(df, ['a(%)', 'b', 'r(%)', 'R(%)'], x_tick_interval=5)
# or
# plot_m_vs_period(df, ['a(%)', 'b', 'r(%)', 'R(%)'], custom_x_ticks=[0, 10, 20, 30])


# same as above but does custom period_Q

# Defining the function as provided by the user
def plot_m_vs_period_Q(excel_file_path, y_variables, start_date, end_date, x_tick_interval=None, custom_x_ticks=None):
    # Load data from Excel file
    dataframe = pd.read_excel(excel_file_path)

    # Convert 'Period_Q' to datetime and filter data
    dataframe['Period_Q'] = pd.to_datetime(dataframe['Period_Q'])
    mask = (dataframe['Period_Q'] >= pd.to_datetime(start_date)) & (dataframe['Period_Q'] <= pd.to_datetime(end_date))
    filtered_df = dataframe[mask]

    num_vars = len(y_variables)
    num_rows = (num_vars + 3) // 4  # Calculate number of rows needed
    num_cols = min(num_vars, 4)     # Maximum of 4 columns

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
    axs = axs.flatten()  # Flatten in case of a single row

    for i, var in enumerate(y_variables):
        if var in filtered_df.columns:
            axs[i].scatter(filtered_df['Period_Q'], filtered_df[var])
            axs[i].plot(filtered_df['Period_Q'], filtered_df[var], linestyle='-', color='blue')
            axs[i].set_xlabel('Period')
            axs[i].set_ylabel(var)
            axs[i].set_title(f'Parameter: {var}')
            
            # Set custom x-axis ticks if provided, otherwise use interval
            if custom_x_ticks is not None:
                axs[i].set_xticks(custom_x_ticks)
            elif x_tick_interval is not None:
                axs[i].set_xticks(pd.date_range(start=pd.to_datetime(start_date), end=pd.to_datetime(end_date), periods=x_tick_interval))

            axs[i].grid(True)
        else:
            print(f"Parameter '{var}' not found in DataFrame.")

    # Hide any unused subplots
    for j in range(i + 1, num_rows * num_cols):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()


# Example usage:
# plot_m_vs_period_Q("path_to_your_excel_file.xlsx", ['C/Y_Q', 'I/Y_Q', 'G/Y_Q', 'NX/Y_Q', 'Y_actual_Q'], '01/01/2016', '12/31/2022', x_tick_interval=5)







# ------ - -    11 CH IS model -- - - - - - -  -- - - -

# this does code does the scatter plots and functions (investment, Consumption, & Imports)
# code is not link just to is func, but just needs an x and y varibale
def IS_Model_Plot(file_path, x_column, y_column, title):
    # Read data from the Excel file
    data = pd.read_excel(file_path)

    # Remove NaN values
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_column, y_column])

    # Extract the relevant columns
    x_data = data[x_column]
    y_data = data[y_column]

    # Fit the regression line
    X = sm.add_constant(x_data)
    model = sm.OLS(y_data, X).fit()
    predictions = model.predict(X)
    slope = model.params[x_column]
    intercept = model.params['const']
    r_squared = model.rsquared

    # Creating scatter plot 
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x_data, y=y_data)
    sns.lineplot(x=x_data, y=predictions, color='red')

    # Linear equation and R² 
    plt.text(x=x_data.max() * 0.95, y=y_data.max() * 0.95, 
             s=f'Y = {intercept:.4f} + {slope:.4f}X\nR² = {r_squared:.4f}', 
             color='blue', fontsize=12, ha='right', va='top')

    plt.title(title)
    plt.xlabel(x_data.name)
    plt.ylabel(y_data.name)
    plt.grid(True)
    plt.show()

# Example usage
# file_path = "C:/Users/eduar/OneDrive - BYU-Idaho/Desktop/Coding/Macro/macrodata.xlsx"
# IS_Model_Plot(file_path, 'R-r_Y', 'I/Y_Y', "Investment Function")


# this does code does multiple scatter plots and functions (investment, Consumption, & Imports)
# code is not link just to is func, but just needs an x and y varibale
def IS_Model_Plot_m(file_path, x_column, y_column, title, plot_index):
    if plot_index % 3 == 1:
        plt.figure(figsize=(20, 6))
    data = pd.read_excel(file_path)
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_column, y_column])
    x_data = data[x_column]
    y_data = data[y_column]
    X = sm.add_constant(x_data)
    model = sm.OLS(y_data, X).fit()
    predictions = model.predict(X)
    slope = model.params[x_column]
    intercept = model.params['const']
    r_squared = model.rsquared
    # Creating scatter
    ax = plt.subplot(1, 3, plot_index % 3 or 3)
    sns.scatterplot(x=x_data, y=y_data, ax=ax)
    sns.lineplot(x=x_data, y=predictions, color='red', ax=ax)
    # Linear equation and R² 
    ax.text(x=x_data.max() * 0.95, y=y_data.max() * 0.95, 
            s=f'Y = {intercept:.4f} + {slope:.4f}X\nR² = {r_squared:.4f}', 
            color='blue', fontsize=12, ha='right', va='top')
    ax.set_title(title)
    ax.set_xlabel(x_data.name)
    ax.set_ylabel(y_data.name)
    ax.grid(True)

    # Show the figure after every third plot
    if plot_index % 3 == 0:
        plt.tight_layout()
        plt.show()

# Example usage
# file_path = "C:/Users/eduar/OneDrive - BYU-Idaho/Desktop/Coding/Macro/macrodata.xlsx"
# IS_Model_Plot_m(file_path, 'R-r_Y', 'I/Y_Y', "Investment Function 1", 1)
# IS_Model_Plot_m(file_path, 'R-r_Y', 'C/Y_Y', "Consumption Function", 2)
# IS_Model_Plot_m(file_path, 'R-r_Y', 'IM/Y_Y', "Imports Function", 3)












def CH13_Short_Run_Output(df, period_start, period_end, variables):
    """
    Plots specified economic indicators for a specified period range from a DataFrame.

    Parameters:
    df (DataFrame): DataFrame containing the data.
    period_start (int): Starting period for the plot.
    period_end (int): Ending period for the plot.
    variables (list): List of column names to plot.
    """
    import matplotlib.pyplot as plt

    # Check if the DataFrame has the necessary columns
    if not all(column in df.columns for column in variables):
        missing_columns = [column for column in variables if column not in df.columns]
        print(f"Missing columns in DataFrame: {', '.join(missing_columns)}")
        return

    # Filtering the DataFrame for the specified period range
    filtered_df = df[df['Period'].between(period_start, period_end)]

    # Plotting
    plt.figure(figsize=(15, 8))

    # Assigning different colors for each variable
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
    for i, var in enumerate(variables):
        plt.plot(filtered_df['Period'], filtered_df[var], label=var, color=colors[i % len(colors)])

    # Adding title and labels
    plt.title('Economic Indicators for Periods {} to {}'.format(period_start, period_end))
    plt.xlabel('Period')
    plt.ylabel('Value')
    plt.legend()

    # Removing gridlines
    plt.grid(True)

    # Display the plot (un mark the next two lines for normal output)
    # plt.tight_layout()
    # plt.show()

    # the following is for custom ticks. need to just add to full def later
    # Example custom ticks: 
    custom_ticks = [(j, str(j+15)) for j in range(20)]  # Generate custom ticks from 16 to 35
    plt.xticks(*zip(*custom_ticks))
    plt.show()

# Example usage of the function
# CH13_Short_Run_Output(df, 0, 20, ['Column1', 'Column2', 'Column3'])






