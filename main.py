import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.random import default_rng

# 1: a,b,c
def read(name):
    df = pd.read_csv(name)
    return df

def group(df,column):
    grouped = df.groupby(column).mean()
    return grouped

def fill_empty(df):
    filled=df.fillna(value='N\A')
    return filled

def drop(df):
    dropped = df.dropna()
    return dropped

def merger (df1,df2):
    merged = pd.merge(df1, df2, on='Identifier')
    return merged


df_email = read('email.csv')
df_user = read('username.csv')

grouped_email = group(df_email,'Identifier')
grouped_user = group(df_email,'Identifier')

filled_email = fill_empty(df_email)
filled_user = fill_empty(df_user)

dropped_email = drop(df_email)
dropped_user = drop (df_user)

merged_df = merger(df_email,df_user)
merged_df.to_csv('merged_df.csv')

# 2: a,b
def pivoter(df,ind,val,aggf,col=None):
    df_pivoted = pd.pivot_table(df,
                                index=ind,
                                columns=col,
                                values=val,
                                aggfunc=aggf)
    return df_pivoted


df_sales = read('sales.csv')
df_data = read('data.csv')

pivoted_sales = pivoter (df_sales,['Rep','Manager','Product'],['Price','Quantity'],[np.sum])
pivoted_data = pivoter (df_data,'Date','Sales','sum','Product')

pivoted_sales.to_csv('pivoted_sales.csv')
pivoted_data.to_csv('pivoted_data.csv')


# 3: a
df = read('sales.csv')
df.plot(kind='bar', x='Rep',y='Price')
plt.show()

# 3: b
data1 = np.random.normal(50, 10, 100)
plt.hist(data1, bins=15)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Data')
plt.show()

# 3: c
data2 = np.random.normal(60, 15, 100)

fig, ax = plt.subplots()
ax.plot(data1, label='Dataset 1')
ax.plot(data2, label='Dataset 2')
ax.set_xlabel('Index')
ax.set_ylabel('Value')
ax.set_title('Comparison of Two Datasets')
ax.legend()
plt.show()

# 3: d
fig, ax = plt.subplots()
x = np.linspace(-np.pi*2, np.pi*2, 100)
y = np.sin(x)
z = np.cos(x)

sin_line, = ax.plot(x, y, color='b', label='sin')
cos_line, = ax.plot(x, z, color='g', label='cos')

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Plot of the Cos and Sin Functions')
ax.legend()
plt.show()

# 3: e
def update(frame):
    x_shifted = x + frame/10
    sin_line.set_ydata(np.sin(x_shifted))
    cos_line.set_ydata(np.cos(x_shifted))
    return sin_line, cos_line

fig, ax = plt.subplots()
x = np.linspace(-np.pi*2, np.pi*2, 100)
y = np.sin(x)
z = np.cos(x)

sin_line, = ax.plot(x, y, color='b', label='sin')
cos_line, = ax.plot(x, z, color='g', label='cos')

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Plot of the Cos and Sin Functions')
ax.legend()
ani = FuncAnimation(fig, update, frames=1000, interval=10, blit=True)
plt.show()


# 4
df_climate = read('climate.csv')
print(df_climate.head())


# 5.a
df_sales = read('sales.csv')
condition = df_sales['Status'] == 'presented'
filtered_df = df_sales[condition]
ranked_df = filtered_df.sort_values (by=['Price'])


 # 5: b
df_climate = read('climate.csv')
condition = (df_climate['cri_score']> 100) & (df_climate['fatalities_total']<11)
filtered_climate_df = df_climate[condition]


# 5: c
df_cars=read('cars.csv')
condition = (df_cars['MPG']>25) & (df_cars['Displacement'] / df_cars['Cylinders'] <=40)
df_cars_filtered = df_cars[condition]
ranked_cars_df = df_cars_filtered.sort_values (by=['Car'])
first_50 = ranked_cars_df.head(50)

# 6
def mean_dev_max(data):
    m = np.mean(data)
    std = np.std(data)
    max_value = np.max(data)
    return m, std, max_value

cars = np.genfromtxt('cars.csv', delimiter=',', usecols=(1), skip_header=1, usemask=True)
mean, dev, max = mean_dev_max(cars)
print("Mean:", mean, '\n',''"Standard deviation:", dev,'\n',"Max value:", max)

# 7
matrix1 = default_rng().random((5,5))
matrix2 = default_rng().random((5,5))

add_matrix = np.add(matrix1,matrix2)
sub_matrix = np.subtract(matrix1,matrix2)
mul_matrix = np.multiply(matrix1,matrix2)
trans_matrix = np.transpose(matrix1)

print("Multiplication of matrices:","\n", mul_matrix)
print("Subtraction of matrices:","\n", sub_matrix)
print("Addition of matrices:","\n", add_matrix)
print('Transposed matrix:','\n',trans_matrix)
