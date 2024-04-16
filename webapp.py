from flask import Flask, render_template, request
import pandas as pd
import numpy as np 
import mpld3
import keras
from sklearn.preprocessing import RobustScaler
scaler1=RobustScaler()
scaler2=RobustScaler()
model=keras.models.load_model(r"C:\Users\rrscnpc-14\Desktop\floodForecasting\model\model.h5")
from pandas import DataFrame
from pandas import concat
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('Agg')
from mpld3 import plugins

app = Flask(__name__)

# Sample DataFrame creation (you should replace this with your actual DataFrame)
data = {'Year': [2023, 2023, 2023, 2023, 2023],
        'Month': [1, 1, 1, 1, 1],
        'Day': [10, 11, 12, 13, 14],
        'Value1': [10, 20, 30, 40, 50],
        'Value2': [15, 25, 35, 45, 55]}

df = pd.read_csv(r"C:\Users\rrscnpc-14\Desktop\floodForecasting\bagmati_dataset_clean.csv",index_col=False)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the user-input values for date and other fields
        year = int(request.form['year'])
        month = int(request.form['month'])
        day = int(request.form['day'])

        # Filter the DataFrame based on the provided date and the two days behind it
        selected_rows = df[
            (df['Year'] == year) & 
            (df['Month'] == month) & 
            (df['Day'] >= day - 2) & 
            (df['Day'] <= day)].reset_index(drop=True)
        
        # Display the selected rows (for testing purposes)
        df_supervised = pd.DataFrame()
        df_supervised = series_to_supervised(selected_rows.drop(columns=['Time'], axis=1), n_in=2, n_out=1)
       # print(df_supervised[['var18(t)','var17(t)']])
        xtest=df_supervised[features]
        ytest=df_supervised[labels]
        #print(xtest.shape,ytest.shape)
        x_scaled=scaler1.fit_transform(xtest)
        y_scaled=scaler2.fit_transform(ytest)
        x_scaled_arr=x_scaled.reshape(-1,1,51)
        y_scaled_arr=y_scaled.reshape(-1,1,1)
        pred=model.predict(x_scaled_arr)
        pred_re=pred.reshape(1,1)
        print(scaler2.inverse_transform(pred_re))
        global prediction
        prediction=scaler2.inverse_transform(pred_re)
        print(ytest.values)
        global actual
        actual=ytest.values
        createmonthdataset(month,year)
    return render_template('ffs.html',prediction=float(prediction),actual=float(actual))



def render_preds():
    return render_template('ffs.html')
    print(prediction,actual)


features = ['var1(t-2)', 'var2(t-2)', 'var3(t-2)', 'var4(t-2)', 'var5(t-2)',
            'var6(t-2)', 'var7(t-2)', 'var8(t-2)', 'var9(t-2)', 'var10(t-2)',
            'var11(t-2)', 'var12(t-2)', 'var13(t-2)', 'var14(t-2)', 'var15(t-2)',
            'var16(t-2)', 'var17(t-2)', 'var18(t-2)', 'var1(t-1)', 'var2(t-1)',
            'var3(t-1)', 'var4(t-1)', 'var5(t-1)', 'var6(t-1)', 'var7(t-1)',
            'var8(t-1)', 'var9(t-1)', 'var10(t-1)', 'var11(t-1)', 'var12(t-1)',
            'var13(t-1)', 'var14(t-1)', 'var15(t-1)', 'var16(t-1)', 'var17(t-1)',
            'var18(t-1)', 'var1(t)', 'var2(t)', 'var3(t)', 'var4(t)', 'var5(t)',
            'var6(t)', 'var7(t)', 'var8(t)', 'var9(t)', 'var10(t)', 'var11(t)',
            'var12(t)', 'var13(t)', 'var14(t)', 'var15(t)']

labels = ['var17(t)']

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
def plot_animation(predictions, actual):
    fig, ax = plt.subplots()
    line_actual= ax.plot([], [], label='Actual')
    line_predicted = ax.plot([], [], label='Predicted')

    # Add legend
    ax.legend()

    # Function to update the plot for each frame
    def update(frame):
        line_actual.set_data(monthly_supervised.index[:frame], prediction[:frame])
        line_predicted.set_data(monthly_supervised.index[:frame], actual[:frame])

    # Create an animation
    ani = FuncAnimation(fig, update, frames=len(monthly_supervised), blit=False)

    # Add a D3-based plugin to the plot
    plugins.connect(fig, plugins.Reset(), plugins.BoxZoom(), plugins.Zoom())

    # Convert the Matplotlib figure with animation to an MPLD3 HTML representation
    global mpld3_html 
    mpld3_html= mpld3.fig_to_html(fig)

    return render_template('ffs.html', mpld3_html=mpld3_html)



def createmonthdataset(month, year):
    monthlydataset=pd.DataFrame()
    monthlydataset=df[(df['Year'] == year) & (df['Month'] == month)]
    global monthly_supervised
    monthly_supervised = series_to_supervised(monthlydataset.drop(columns=['Time'], axis=1), n_in=2, n_out=1)
    xtestm=monthly_supervised[features]
    ytestm=monthly_supervised[labels]
    #print(xtest.shape,ytest.shape)
    x_scaledm=scaler1.fit_transform(xtestm)
    y_scaledm=scaler2.fit_transform(ytestm)
    x_scaled_arrm=x_scaledm.reshape(-1,1,51)
    y_scaled_arrm=y_scaledm.reshape(-1,1,1)
    predm=model.predict(x_scaled_arrm)
    pred_rem=predm.reshape(-1,1)
    print(scaler2.inverse_transform(pred_rem))
    global predictionm
    predictionm=scaler2.inverse_transform(pred_rem)
    print(ytestm.values)
    global actualm
    actualm=ytestm.values
    plot_animation(prediction,actual)
if __name__ == '__main__':
    app.run(debug=True)
