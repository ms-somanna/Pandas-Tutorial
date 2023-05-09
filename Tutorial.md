# <center>`Pandas Tutorial`</center>
<hr>

<div>
<img src="attachment:pandas_logo.svg" width="50%"/>
</div>

Pandas is a open source python library used for data analysis and manipulation.  
<br>
Features:
- Provides data-structures that can be used to store and manipulate data.
- Two data-structures are provided, namely, `Series` and `Dataframe`.  
<br>

[Homepage of Pandas](https://pandas.pydata.org/)


## `Prerequisites`
Since Pandas is a python library and is built on top of Numpy, you are required to know the basics of python and numpy before you proceed with this tutorial.

## `Installation`
>pip install:  
`pip install pandas`

>Anaconda install:  
`conda install -c anaconda pandas`

## `Initialize`


```python
# Import pandas
import pandas as pd # Importing pandas with alias 'pd'
import numpy as np # numpy offers additional capabilities to pandas.

# Checking Pandas current version
print('Pandas Version: ', pd.__version__)

# Checking Numpy current version
print('Numpy Version: ', np.__version__)
```

    Pandas Version:  2.0.1
    Numpy Version:  1.24.3
    

# `Data Structures`

We'll study two datastructures namely, 'Series' and 'Datafarme'. Though Series is used quite less often compared to Dataframe, many of the row and column operations performed on Dataframes, return a series. This means, such row/column operations are performed at a series level. Hence it becomes necessary to learn series operations first, before moving on to learning operations on Dataframe.  We'll first learn some common operations that can be performed on a series then move on to learning operations on Dataframes.

## Series
A series is basically a 1D array whose values are lebeled for easy access. And all the values are of the same datatype. (see image below)  
Things to note about Series:
- The array in the series is of the type `numpy ndarray`.
- The labels together are called as 'Index'. The index is of the type `pandas.core.indexes.base.Index`
- Since all the values are of the same datatype, Series are called as being _homogeneous_.
<br>
<hr>
<h3>Structure of a Series</h3>
<div>
<img src="attachment:series_disc.svg" width="50%" align="center"/>
</div>
<br>
<hr>


```python
# Check the datatype of array containg the values. Must be a numpy ndarray
type(pd.Series().values)
```




    numpy.ndarray




```python
# Check the datatype of Index
type(pd.Series().index)
```




    pandas.core.indexes.range.RangeIndex



### Usage  
`pd.Series(data=None, index=None, dtype=None, name=None, copy=False)`  
* **Returns:**  
    `Series` object.  
<br>

* **Parameters:**
    - `data`: The input data. A scalar or an iterable, like a list, tupple, dict, 1D array that contains the values to be stored in the series.
    - `index`: An 1D iterable containg the labels for each value. If index is not specified, a range index will be used by the series.
    - `dtype`: (string, numpy.dtype) The datatype of the values. If not passed, the dtype of the values will be used, if the values have different dtypes, object dtype will be used.
    - `name`: (Optional string) Name to give to the series. If no name is passed, no name is assigned to the series.
    - `copy`: (Bool) If True is passed, A copy of input data is set as the series values, if False is passed a view of the input data is set as series values. Only affects Series or 1D ndarray input.

### Create a Series
#### Create a Series from a List


```python
# Create a series with data passed as a list of values
pd.Series([1,2,3,4])
```




    0    1
    1    2
    2    3
    3    4
    dtype: int64



In the output above:
- the numbers from 0 to 3 on the left side are the index. Since no index is passed, pandas automatically used a range index (0 to 3).
- the numbers from 1 to 4 on the right side are the values.  


```python
# Assign series to a variable
s = pd.Series([1,2,3,4])
print(s)
```

    0    1
    1    2
    2    3
    3    4
    dtype: int64
    


```python
# You can use pass values of any datatype 
pd.Series([1,'a',True,4.99])
```




    0       1
    1       a
    2    True
    3    4.99
    dtype: object



When values of different dtypes are passed to a series, the series stores these values as object dtype, but when these values are referenced individually from the series, they are referenced with their original dtype.


```python
# Passing values with different dtypes to a series, and checking the dtypes of each of those values
# when they are referenced individually.

s = pd.Series(['a',1,0,True]) # Values passed with different dtypes.
print(s) # Should print dtype as 'object'
vals_types = '\n'.join(['Value: {}, dtype: {}'.
                        format(v, str(type(v)).split("'")[1]) for v in s.values]) # Values and their datatypes
print('\n',vals_types)
```

    0       a
    1       1
    2       0
    3    True
    dtype: object
    
     Value: a, dtype: str
    Value: 1, dtype: int
    Value: 0, dtype: int
    Value: True, dtype: bool
    

You can read more about this datatype ambiguity at https://stackoverflow.com/questions/70718929/pandas-is-series-homogenous


```python
pd.Series(['male','male','female'])
```




    0      male
    1      male
    2    female
    dtype: object



str dtype passed to a series will be held as object dtype in the series.


```python
# Creating Series from Numpy Array
pd.Series(np.array([1,2,3,4]))
```




    0    1
    1    2
    2    3
    3    4
    dtype: int64




```python
# Creating Series from another Series.
s = pd.Series(['a', 'b', 'c'])
pd.Series(s)
```




    0    a
    1    b
    2    c
    dtype: object



<div class="alert alert-block alert-warning">
⚠️<b> Warning</b>
<hr>
The <code>copy</code> parameter of the series is set to <code>False</code> by default. This means when the input data passed to a Series is an 1d ndarray object or another Series object, a view of that object is created. Thus modifying the series values will modify the data object that was passed aswell. So care must be taken while using a series or an ndarray as input data to another series. If you want to create a copy instead of a view, set the Series's <code>copy</code> parameter to <code>True</code>.
</div>

#### Create a Series by Passing an Index/Labels


```python
# Pass index/labels
pd.Series([1,2,3,4], index=['a','b','c','d'])
```




    a    1
    b    2
    c    3
    d    4
    dtype: int64




```python
# Each Index label can be of any datatype
pd.Series([1,2,3,4], index=['a', 1, True, None])
```




    a       1
    1       2
    True    3
    None    4
    dtype: int64




```python
#Pandas suports duplicate labels
pd.Series([1,2,3], index=['a','b','b'])
```




    a    1
    b    2
    b    3
    dtype: int64




```python
#Create series with scalar data
pd.Series(1)
```




    0    1
    dtype: int64




```python
# Create series with a list containing one value. Same as creating a series with a scalar.
pd.Series([1])
```




    0    1
    dtype: int64



<div class="alert alert-block alert-info">
🗒<b> Note</b>
<hr>
As you may have noticed in the outputs of the above two cells, creating a series with a scalar and with a list containg single value has the same effect. 
</div>


```python
# With the help of the passed index, you can set the same scalar value as all series values addressed by the index.
pd.Series(3.2, index=[1,2,3,4])
```




    1    3.2
    2    3.2
    3    3.2
    4    3.2
    dtype: float64




```python
# Another way without using an index.
pd.Series([3.2]*4)
```




    0    3.2
    1    3.2
    2    3.2
    3    3.2
    dtype: float64



<div class="alert alert-block alert-info">
🗒<b> Note</b>
<hr>
Creating a list by repeating a [value] by a given number of times is a python trick and not a pandas trick.
    <br>
[val]*n = [val(0), val(1),...val(n-1)] 
</div>


```python
# Passing datatype
pd.Series([-1,0,1,2], dtype=float)
```




    0   -1.0
    1    0.0
    2    1.0
    3    2.0
    dtype: float64




```python
pd.Series([-1,0,1,2], dtype='float32')
```




    0   -1.0
    1    0.0
    2    1.0
    3    2.0
    dtype: float32




```python
# Create a Series from a dict.
pd.Series({'a':1, 'b':2, 'c':3, 'd':4})
# If dict is passed as data to the series, dict keys become index and, dict values become series values.
```




    a    1
    b    2
    c    3
    d    4
    dtype: int64



### Manipulate the Index


```python
# Get the index of a series
s = pd.Series([1,2,3,4], index=['a','b','c','d'])
s.index
```




    Index(['a', 'b', 'c', 'd'], dtype='object')




```python
# Modify the index that was previously assigned
s = pd.Series([1,2,3,4], index=[1,2,3,4])

print(f'Original Series:\n{s}')
s.index = ['a','b','c','d']
print(f'\nSame series with modified index:\n{s}')
```

    Original Series:
    1    1
    2    2
    3    3
    4    4
    dtype: int64
    
    Same series with modified index:
    a    1
    b    2
    c    3
    d    4
    dtype: int64
    

The index object has several useful methods and attributes similar to a python list, like converting the index to a list, fetching index elements using python list indexing, slicing etc.


```python
# Get series index as a list
s.index.to_list()
```




    ['a', 'b', 'c', 'd']




```python
# Get series index as numpy array. Doesn't require numpy import.
np.array(s.index)
```




    array(['a', 'b', 'c', 'd'], dtype=object)




```python
# Get the dtype of the index
s.index[:2].tolist()
```




    ['a', 'b']




```python
# Check if a certain index label exists in series
'a' in s
```




    True



### Manipulate the Values


```python
# Get the values of a series
s = pd.Series([1,2,3,4], index=['a','b','c','d'])
s.values
```




    array([1, 2, 3, 4])




```python
# Convert the value array to a list using the numpy built-in method 'ndarray.to_list()'
s.values.tolist()
# Remember, the list of values of a series is a numpy.ndarray?
```




    [1, 2, 3, 4]




```python
# Get the count of values in the series.
# Count is the number of values in the series.
s.count()
```




    4



4 was returned, since there are 4 values in the series.


```python
# Get the datatype of the series
s = pd.Series([1,2,3])
s.dtype
```




    dtype('int64')




```python
# Convert datatype
s = pd.Series([1,2,3,4]) #Original series
display(s)
s = s.astype('float') # Series with converted dtype. 
display(s)
```


    0    1
    1    2
    2    3
    3    4
    dtype: int64



    0    1.0
    1    2.0
    2    3.0
    3    4.0
    dtype: float64



```python
# Index based sorting:
# Explictly passing index along with data that is a dict, will result in values being sorted according to 
# the passed index.
pd.Series({'a':1, 'b':2, 'c':3, 'd':4}, index=['b','c','d','a'])
```




    b    2
    c    3
    d    4
    a    1
    dtype: int64




```python
# Passing an index value that is not matching any dict key will result in addition of a NaN value
# at the index value location
pd.Series({'a':1, 'b':2, 'c':3, 'd':4}, index=['b','c','d','z'])
```




    b    2.0
    c    3.0
    d    4.0
    z    NaN
    dtype: float64



<div class="alert alert-block alert-info">
🗒<b> Note</b>
    <hr>
NaN (abrivation for <i>Not A Number</i>) is the standard missing data marker used in pandas.
</div>


```python
# Fetching series values with index location
s = pd.Series([1,2,3,4], index=['a','b','c',True])
s[0]
```




    1




```python
# You can also pass the index value to the iloc function of the series.
s.iloc[0]
```




    1




```python
# Fetching series elements with index label, like how you do in fetching a dict value from its key.
s['a']
```




    1




```python
s[[0,2,3]]
# While fetching values from a list of individual index labels, the labels must be passed as a list.
```




    a       1
    c       3
    True    4
    dtype: int64




```python
# Slicing series
# Slicing series is similar to slicing numpy ndarray.

s = pd.Series([1,2,3,4,5], index=['a','b','c','d','e']) # Define Series for slicing
s
```




    a    1
    b    2
    c    3
    d    4
    e    5
    dtype: int64




```python
s[1:3] # Slicing to get values in range [1,3) (1 inclided, 3 not included)
```




    b    2
    c    3
    dtype: int64




```python
s[:3] # Slicing to get values in range [0,3)
```




    a    1
    b    2
    c    3
    dtype: int64




```python
s['a':'c']
```




    a    1
    b    2
    c    3
    dtype: int64




```python
s['c':'a']
```




    Series([], dtype: int64)




```python
s[['a','b','c']]
```




    a    1
    b    2
    c    3
    dtype: int64



### Perform Mathematical Operations on Series


```python
# Like on numpy array, vectorized (element-wise) operations can be performed on series
s + 2
```




    a    3
    b    4
    c    5
    d    6
    e    7
    dtype: int64




```python
# Vectorization enables some quick opereations on the elemtents

# Example 1: Quickly convert all elements to 0s
s - s
```




    a    0
    b    0
    c    0
    d    0
    e    0
    dtype: int64




```python
# Example 2: Quickly convert all elements to 1s
s // s
```




    a    1
    b    1
    c    1
    d    1
    e    1
    dtype: int64




```python
# Example 3: 
s + (s//1)
```




    a     2
    b     4
    c     6
    d     8
    e    10
    dtype: int64




```python
# Example 4: Quickly convert series dtype from int to float
s / 1
```




    a    1.0
    b    2.0
    c    3.0
    d    4.0
    e    5.0
    dtype: float64




```python
np.log2(s) # Does element-wise operations on the series values and returns a series with operated values.
```




    a    0.000000
    b    1.000000
    c    1.584963
    d    2.000000
    e    2.321928
    dtype: float64




```python
# If the series is not of object dtype (as in this case), you can vectorically
# perform string operations on each non-string value by converting them to strings using
# `series.astype(str)'
s.astype(str) + ' kg'
```




    a    1 kg
    b    2 kg
    c    3 kg
    d    4 kg
    e    5 kg
    dtype: object




```python
s['a':'c'] + s['a':'c']
# Note: elements are automatically alligned based on the order of the label.
# When performing operations on elements specified by their indeses, the union of those indeses
# will become the index of the resulting series.
# (a, b, c) ∪ (a, b, c) = (a,b,c)
```




    a    2
    b    4
    c    6
    dtype: int64




```python
s['a':'c'] + s['c':'d'] # (a, b, c) ∪ (c, d) = (a,b,c,d)
```




    a    NaN
    b    NaN
    c    6.0
    d    NaN
    dtype: float64




```python
# Conditional operations
s<3
```




    a     True
    b     True
    c    False
    d    False
    e    False
    dtype: bool




```python
s == 4
```




    a    False
    b    False
    c    False
    d     True
    e    False
    dtype: bool




```python
# Fetch values satisfying a condition
s[s<3]
```




    a    1
    b    2
    dtype: int64




```python
s[s==3]
```




    c    3
    dtype: int64




```python
# Get the mean of the values in the seriess
s.mean()
```




    3.0




```python
# Median
s.median()
```




    3.0




```python
# Get the sum of all values
s.sum()
```




    15




```python
# Name the Series
s = pd.Series([1,2,3], index=['a','b','c'], name='ShortSeries')
s
```




    a    1
    b    2
    c    3
    Name: ShortSeries, dtype: int64




```python
# Rename the series
s = s.rename('Series S')
s
```




    a    1
    b    2
    c    3
    Name: Series S, dtype: int64




```python
# Create a copy of the Series 
r = s.copy()
r
```




    a    1
    b    2
    c    3
    Name: Series S, dtype: int64




```python
# Create a view of the Series.
v = s.view()
v
```




    a    1
    b    2
    c    3
    Name: Series S, dtype: int64




```python
# Get information about series
s.info()
```

    <class 'pandas.core.series.Series'>
    Index: 3 entries, a to c
    Series name: Series S
    Non-Null Count  Dtype
    --------------  -----
    3 non-null      int64
    dtypes: int64(1)
    memory usage: 156.0+ bytes
    


```python
# Show a number of entries from top to down
s.head(2)
```




    a    1
    b    2
    Name: Series S, dtype: int64




```python
# Show a number of entries from bottom to up
s.tail(2)
```




    b    2
    c    3
    Name: Series S, dtype: int64




```python
# Drop null values
s = pd.Series([1, 2, None, 4, np.nan]) # Create series with a null value
display(s)
s = s.dropna() #Use dropna method to drop all null values.
s
```


    0    1.0
    1    2.0
    2    NaN
    3    4.0
    4    NaN
    dtype: float64





    0    1.0
    1    2.0
    3    4.0
    dtype: float64




```python
# Check if each value is null in a series
s = pd.Series([1, 2, None, 4, np.nan]) # Create series with a null value
s.isna() # Returns a series with bool values, True in place of null values, False everywhere else.
```




    0    False
    1    False
    2     True
    3    False
    4     True
    dtype: bool




```python
# Count the number of null values in the series
s.isna().sum()
# In python, when performing arethematic operations on boolian values,
# True is treated as 1 and False is treated as 0.
# So, False + False + True + False + True = 0 + 0 + 1 + 0 + 1 = 2
```




    2



<div class="alert alert-block alert-info">
🗒<b> Note</b>
    <hr>
While calculating mean, median and some other scores, the the null values are ignored.
</div>


```python
# Inplace changes: Instead of performing reassignment of series
# in order to make any changes to the series (E.g s = s.rename('name2')),
# you can pass a True to the papameter called 'inplace' present in methods
# (like 'rename') that make changes to a series.
s = pd.Series([1,None,3,None,5], name='Old name') # Create series with passing a name
display(s)
s.rename('New name', inplace=True) # Rename with inplace=True
s
```


    0    1.0
    1    NaN
    2    3.0
    3    NaN
    4    5.0
    Name: Old name, dtype: float64





    0    1.0
    1    NaN
    2    3.0
    3    NaN
    4    5.0
    Name: New name, dtype: float64




```python
s.dropna(inplace=True)
s
```




    0    1.0
    2    3.0
    4    5.0
    Name: New name, dtype: float64




```python
# Join two series into one
s = pd.Series([1,2,3], index=['a','b','c'])
r = pd.Series([4,5,6], index=['d','e','f'])
t = pd.concat([s,r])
t
```




    a    1
    b    2
    c    3
    d    4
    e    5
    f    6
    dtype: int64



----------------------------------------------------

## Dataframe
Dataframe is a spreadsheet like data-structure, with labeled columns and indexed rows. (See image below)  
- Since it is like a spreadsheet, it is two dimentional with rows and columns.
- If row and column labels are not be passed to the dataframe, range index will be used as the row and column labels.
- Each dimention is called as an _axis_. The axis along the columns is called "Axis 0", and the axis along the rows is called "Axis 1".
- Each value in the datafrem is located by it's row and column label.
- More comonly used pandas datastructure compared to series.  


### Structure of a Dataframe
<hr>
<div>
<img src="attachment:dataframe_disc-2.svg" width="70%" align="center"/>
</div>
<hr>

### Usage 
`pd.DataFrame(data=None, index=None, columns=None)`  
- data: ndarray (structured or homogeneous), Iterable, dict, or DataFrame.
- index: (Optional) An 1D iterable containg the labels for each element. If index is not passed, a range index will be used.
- columns: (Optional) 1D iterable containing the column labels for each column. If not passed, and the data itself contains column labels, thos labels will be used, or else a range index will be used.
### Create and Manipulate a Dataframe


```python
# Create dataframe from a dict of series.
d = {
    'Col 1': pd.Series([1,2,3,4]),
    'Col 2': pd.Series([5,6,7,8])
}
pd.DataFrame(d)
# Note:
#    1. If columns are not explicitly passed, the dict keys will be used as the column labels
#       and the columns will be arranged accroding to the dict key order.
#    2. If index is not explicitly passed, range index will be used as index.
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Col 1</th>
      <th>Col 2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Order of column arrangement can be changed and set accordingly by passing columns in the desired order.
pd.DataFrame(d, columns=['Col 2', 'Col 1'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Col 2</th>
      <th>Col 1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create dataframe with custom Index.
# By passing index, a row is added for each index label and the corresponding
# series values are added as the row values.

d = {
    'Col 1': pd.Series([1,2,3,4], index=['a','b','c','d']),
    'Col 2': pd.Series([5,6,7,8], index=['a','b','c','e'])
}
pd.DataFrame(d)
#Note: The index of bothe series are identical.
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Col 1</th>
      <th>Col 2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>2.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>c</th>
      <td>3.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>d</th>
      <td>4.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>e</th>
      <td>NaN</td>
      <td>8.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Order of row arrangement can be changed and set accordingly by passing index in the desired order.
pd.DataFrame(d, index=['c','a','d'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Col 1</th>
      <th>Col 2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>c</th>
      <td>3</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>a</th>
      <td>1</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>d</th>
      <td>4</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.DataFrame(d, index=['a','d','f','b'], columns=['Col 2', 'Col 1', 'Col 3'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Col 2</th>
      <th>Col 1</th>
      <th>Col 3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>5.0</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>f</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>b</th>
      <td>6.0</td>
      <td>2.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Get dataframe row index
d = {
    'Col 1': pd.Series([1,2,3,4], index=['a','b','c','d']),
    'Col 2': pd.Series([5,6,7,8], index=['a','b','c','d'])
}
df = pd.DataFrame(d)
df.index
```




    Index(['a', 'b', 'c', 'd'], dtype='object')




```python
# Get the dataframe columns
df.columns
```




    Index(['Col 1', 'Col 2'], dtype='object')




```python
# Get the dataframe values
df.values
```




    array([[1, 5],
           [2, 6],
           [3, 7],
           [4, 8]])




```python
# Get value count
df.size
# datafrem size is the number of values present in the datfarme.
```




    8



<div class="alert alert-block alert-info">
🗒<b> Note</b>
    <hr>
    Many functionalities (methods and atrributes) of series are also avilable for Dataframe, such as, index, values, mean, median, index based slicing etc. What functionalities are available to both series and dataframe and what functionalities are not, can be determined by common-sense.
</div>


```python
# Create dataframe from a dict of numpy 1D array or a list.
d = {
    'Col 1': [1,2,3,4],
    'Col 2': np.array([5,6,7,8])
}
pd.DataFrame(d)
# Note: since no index was passed range index was used.
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Col 1</th>
      <th>Col 2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create dataframe from a dict of numpy 1D array or a list and pass index.
d = {
    'Col 1': [1,2,3,4],
    'Col 2': np.array([5,6,7,8])
}
pd.DataFrame(d, index=['a','b','c','d'])
#Note index length must be the same as the length of the data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Col 1</th>
      <th>Col 2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>b</th>
      <td>2</td>
      <td>6</td>
    </tr>
    <tr>
      <th>c</th>
      <td>3</td>
      <td>7</td>
    </tr>
    <tr>
      <th>d</th>
      <td>4</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create dataframe from a series
s = pd.Series([1,2,3,4])
pd.DataFrame(s)
# Note: since no index was passed range index was used.
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
# If name and/or index is passed to the series, the name will be set as column label
# and the series index will be set as index of the dataframe.
s = pd.Series([1,2,3],index=['a','b','c'], name='Col')
pd.DataFrame(s)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Col</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1</td>
    </tr>
    <tr>
      <th>b</th>
      <td>2</td>
    </tr>
    <tr>
      <th>c</th>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



### Manipulate the Columns and Rows


```python
# Getting column.
# When a column is fetched from a dataframe, a series containing the column is returned.

index = pd.Index(['a','b','c','d'])
d = {
    'Col 1': pd.Series([1,2,3,4], index=index),
    'Col 2': pd.Series([5,6,7,8], index=index),
    'Col 3': pd.Series([9,10,11,12], index=index),
    'Col 4': pd.Series([13,14,15,16], index=index),
    'Col 5': pd.Series([17,18,19,20], index=index)
}

df = pd.DataFrame(d)

df['Col 1']
```




    a    1
    b    2
    c    3
    d    4
    Name: Col 1, dtype: int64



Note: Since the fetched column of a dataframe is a series, all methods and operations of a series (some discussed in the series section) can be applied to the fetched column.


```python
# Fetch multiple columns
df[['Col 1', 'Col 2']] # Returns a dataframe
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Col 1</th>
      <th>Col 2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>b</th>
      <td>2</td>
      <td>6</td>
    </tr>
    <tr>
      <th>c</th>
      <td>3</td>
      <td>7</td>
    </tr>
    <tr>
      <th>d</th>
      <td>4</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Operate on columns
df['Col 1'] + df['Col 2']
```




    a     6
    b     8
    c    10
    d    12
    dtype: int64




```python
df['Col 1'] / 1
```




    a    1.0
    b    2.0
    c    3.0
    d    4.0
    Name: Col 1, dtype: float64




```python
df['Col 2'] - df['Col 2']
```




    a    0
    b    0
    c    0
    d    0
    Name: Col 2, dtype: int64




```python
# Edit a column
df['Col 1'] = [1,1,1,1]
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Col 1</th>
      <th>Col 2</th>
      <th>Col 3</th>
      <th>Col 4</th>
      <th>Col 5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1</td>
      <td>5</td>
      <td>9</td>
      <td>13</td>
      <td>17</td>
    </tr>
    <tr>
      <th>b</th>
      <td>1</td>
      <td>6</td>
      <td>10</td>
      <td>14</td>
      <td>18</td>
    </tr>
    <tr>
      <th>c</th>
      <td>1</td>
      <td>7</td>
      <td>11</td>
      <td>15</td>
      <td>19</td>
    </tr>
    <tr>
      <th>d</th>
      <td>1</td>
      <td>8</td>
      <td>12</td>
      <td>16</td>
      <td>20</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['Col 2'] = df['Col 1'] - df['Col 3']
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Col 1</th>
      <th>Col 2</th>
      <th>Col 3</th>
      <th>Col 4</th>
      <th>Col 5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1</td>
      <td>-8</td>
      <td>9</td>
      <td>13</td>
      <td>17</td>
    </tr>
    <tr>
      <th>b</th>
      <td>1</td>
      <td>-9</td>
      <td>10</td>
      <td>14</td>
      <td>18</td>
    </tr>
    <tr>
      <th>c</th>
      <td>1</td>
      <td>-10</td>
      <td>11</td>
      <td>15</td>
      <td>19</td>
    </tr>
    <tr>
      <th>d</th>
      <td>1</td>
      <td>-11</td>
      <td>12</td>
      <td>16</td>
      <td>20</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Delete a column
del df['Col 3']
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Col 1</th>
      <th>Col 2</th>
      <th>Col 4</th>
      <th>Col 5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1</td>
      <td>-8</td>
      <td>13</td>
      <td>17</td>
    </tr>
    <tr>
      <th>b</th>
      <td>1</td>
      <td>-9</td>
      <td>14</td>
      <td>18</td>
    </tr>
    <tr>
      <th>c</th>
      <td>1</td>
      <td>-10</td>
      <td>15</td>
      <td>19</td>
    </tr>
    <tr>
      <th>d</th>
      <td>1</td>
      <td>-11</td>
      <td>16</td>
      <td>20</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Delete a column, and return the deleted column as a series
df.pop('Col 5')
```




    a    17
    b    18
    c    19
    d    20
    Name: Col 5, dtype: int64




```python
# Add a new column
df['Col A'] = ['a','b','c','d']
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Col 1</th>
      <th>Col 2</th>
      <th>Col 4</th>
      <th>Col A</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1</td>
      <td>-8</td>
      <td>13</td>
      <td>a</td>
    </tr>
    <tr>
      <th>b</th>
      <td>1</td>
      <td>-9</td>
      <td>14</td>
      <td>b</td>
    </tr>
    <tr>
      <th>c</th>
      <td>1</td>
      <td>-10</td>
      <td>15</td>
      <td>c</td>
    </tr>
    <tr>
      <th>d</th>
      <td>1</td>
      <td>-11</td>
      <td>16</td>
      <td>d</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['Col 2 Flag'] = df['Col 2'] > -10
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Col 1</th>
      <th>Col 2</th>
      <th>Col 4</th>
      <th>Col A</th>
      <th>Col 2 Flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1</td>
      <td>-8</td>
      <td>13</td>
      <td>a</td>
      <td>True</td>
    </tr>
    <tr>
      <th>b</th>
      <td>1</td>
      <td>-9</td>
      <td>14</td>
      <td>b</td>
      <td>True</td>
    </tr>
    <tr>
      <th>c</th>
      <td>1</td>
      <td>-10</td>
      <td>15</td>
      <td>c</td>
      <td>False</td>
    </tr>
    <tr>
      <th>d</th>
      <td>1</td>
      <td>-11</td>
      <td>16</td>
      <td>d</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Add a new column with scalar value
df['Col Scalars'] = 5
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Col 1</th>
      <th>Col 2</th>
      <th>Col 4</th>
      <th>Col A</th>
      <th>Col 2 Flag</th>
      <th>Col Scalars</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1</td>
      <td>-8</td>
      <td>13</td>
      <td>a</td>
      <td>True</td>
      <td>5</td>
    </tr>
    <tr>
      <th>b</th>
      <td>1</td>
      <td>-9</td>
      <td>14</td>
      <td>b</td>
      <td>True</td>
      <td>5</td>
    </tr>
    <tr>
      <th>c</th>
      <td>1</td>
      <td>-10</td>
      <td>15</td>
      <td>c</td>
      <td>False</td>
      <td>5</td>
    </tr>
    <tr>
      <th>d</th>
      <td>1</td>
      <td>-11</td>
      <td>16</td>
      <td>d</td>
      <td>False</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# When inserting a series as a column, if the series index doesn't
# match the dataframe index, nulls will be added at locations of mismatched index labels.   
df['Col 5'] = pd.Series([1.1,2.2], index=['a','b'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Col 1</th>
      <th>Col 2</th>
      <th>Col 4</th>
      <th>Col A</th>
      <th>Col 2 Flag</th>
      <th>Col Scalars</th>
      <th>Col 5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1</td>
      <td>-8</td>
      <td>13</td>
      <td>a</td>
      <td>True</td>
      <td>5</td>
      <td>1.1</td>
    </tr>
    <tr>
      <th>b</th>
      <td>1</td>
      <td>-9</td>
      <td>14</td>
      <td>b</td>
      <td>True</td>
      <td>5</td>
      <td>2.2</td>
    </tr>
    <tr>
      <th>c</th>
      <td>1</td>
      <td>-10</td>
      <td>15</td>
      <td>c</td>
      <td>False</td>
      <td>5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>d</th>
      <td>1</td>
      <td>-11</td>
      <td>16</td>
      <td>d</td>
      <td>False</td>
      <td>5</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['Col 6'] = df['Col 1'][:2]
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Col 1</th>
      <th>Col 2</th>
      <th>Col 4</th>
      <th>Col A</th>
      <th>Col 2 Flag</th>
      <th>Col Scalars</th>
      <th>Col 5</th>
      <th>Col 6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1</td>
      <td>-8</td>
      <td>13</td>
      <td>a</td>
      <td>True</td>
      <td>5</td>
      <td>1.1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>1</td>
      <td>-9</td>
      <td>14</td>
      <td>b</td>
      <td>True</td>
      <td>5</td>
      <td>2.2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>c</th>
      <td>1</td>
      <td>-10</td>
      <td>15</td>
      <td>c</td>
      <td>False</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>d</th>
      <td>1</td>
      <td>-11</td>
      <td>16</td>
      <td>d</td>
      <td>False</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Inserting a new column at a peticular location
df.insert(1,'Col1_copy', df['Col 1'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Col 1</th>
      <th>Col1_copy</th>
      <th>Col 2</th>
      <th>Col 4</th>
      <th>Col A</th>
      <th>Col 2 Flag</th>
      <th>Col Scalars</th>
      <th>Col 5</th>
      <th>Col 6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1</td>
      <td>1</td>
      <td>-8</td>
      <td>13</td>
      <td>a</td>
      <td>True</td>
      <td>5</td>
      <td>1.1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>1</td>
      <td>1</td>
      <td>-9</td>
      <td>14</td>
      <td>b</td>
      <td>True</td>
      <td>5</td>
      <td>2.2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>c</th>
      <td>1</td>
      <td>1</td>
      <td>-10</td>
      <td>15</td>
      <td>c</td>
      <td>False</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>d</th>
      <td>1</td>
      <td>1</td>
      <td>-11</td>
      <td>16</td>
      <td>d</td>
      <td>False</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Get a new dataframe cpoied from an old one with new and existing columns.
# The old dataframe remains unchanged.

df_new = df.assign(New_Col = df['Col 2'])
df_new
# Note: The column name passed into assign should not have any white spaces.
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Col 1</th>
      <th>Col1_copy</th>
      <th>Col 2</th>
      <th>Col 4</th>
      <th>Col A</th>
      <th>Col 2 Flag</th>
      <th>Col Scalars</th>
      <th>Col 5</th>
      <th>Col 6</th>
      <th>New_Col</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1</td>
      <td>1</td>
      <td>-8</td>
      <td>13</td>
      <td>a</td>
      <td>True</td>
      <td>5</td>
      <td>1.1</td>
      <td>1.0</td>
      <td>-8</td>
    </tr>
    <tr>
      <th>b</th>
      <td>1</td>
      <td>1</td>
      <td>-9</td>
      <td>14</td>
      <td>b</td>
      <td>True</td>
      <td>5</td>
      <td>2.2</td>
      <td>1.0</td>
      <td>-9</td>
    </tr>
    <tr>
      <th>c</th>
      <td>1</td>
      <td>1</td>
      <td>-10</td>
      <td>15</td>
      <td>c</td>
      <td>False</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-10</td>
    </tr>
    <tr>
      <th>d</th>
      <td>1</td>
      <td>1</td>
      <td>-11</td>
      <td>16</td>
      <td>d</td>
      <td>False</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-11</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Get a row by index label.
# Will return a series containg the dataframe row as values and Columns as index.
df.loc['a']
```




    Col 1             1
    Col1_copy         1
    Col 2            -8
    Col 4            13
    Col A             a
    Col 2 Flag     True
    Col Scalars       5
    Col 5           1.1
    Col 6           1.0
    Name: a, dtype: object



Note: Since the fetched row of a dataframe is a series, all methods and operations of a series (some discussed in the series section) can be applied to the fetched row. 


```python
# Get a row by index by integer location
df.iloc[0]
```




    Col 1             1
    Col1_copy         1
    Col 2            -8
    Col 4            13
    Col A             a
    Col 2 Flag     True
    Col Scalars       5
    Col 5           1.1
    Col 6           1.0
    Name: a, dtype: object




```python
# Slice rows
df[:3]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Col 1</th>
      <th>Col1_copy</th>
      <th>Col 2</th>
      <th>Col 4</th>
      <th>Col A</th>
      <th>Col 2 Flag</th>
      <th>Col Scalars</th>
      <th>Col 5</th>
      <th>Col 6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1</td>
      <td>1</td>
      <td>-8</td>
      <td>13</td>
      <td>a</td>
      <td>True</td>
      <td>5</td>
      <td>1.1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>1</td>
      <td>1</td>
      <td>-9</td>
      <td>14</td>
      <td>b</td>
      <td>True</td>
      <td>5</td>
      <td>2.2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>c</th>
      <td>1</td>
      <td>1</td>
      <td>-10</td>
      <td>15</td>
      <td>c</td>
      <td>False</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Get rows using condition.
df[df['Col 4'] < 15]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Col 1</th>
      <th>Col1_copy</th>
      <th>Col 2</th>
      <th>Col 4</th>
      <th>Col A</th>
      <th>Col 2 Flag</th>
      <th>Col Scalars</th>
      <th>Col 5</th>
      <th>Col 6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1</td>
      <td>1</td>
      <td>-8</td>
      <td>13</td>
      <td>a</td>
      <td>True</td>
      <td>5</td>
      <td>1.1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>1</td>
      <td>1</td>
      <td>-9</td>
      <td>14</td>
      <td>b</td>
      <td>True</td>
      <td>5</td>
      <td>2.2</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df['Col 2'] == -8]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Col 1</th>
      <th>Col1_copy</th>
      <th>Col 2</th>
      <th>Col 4</th>
      <th>Col A</th>
      <th>Col 2 Flag</th>
      <th>Col Scalars</th>
      <th>Col 5</th>
      <th>Col 6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1</td>
      <td>1</td>
      <td>-8</td>
      <td>13</td>
      <td>a</td>
      <td>True</td>
      <td>5</td>
      <td>1.1</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Edit a row
df.loc['a'] = [100,1,-100,20,'b',False,-5,10,1]
```


```python
# Insert a new row below the exiting last row.
df.loc['e'] = ['ff','ff',-3,'dd','e',True,10,12,12]
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Col 1</th>
      <th>Col1_copy</th>
      <th>Col 2</th>
      <th>Col 4</th>
      <th>Col A</th>
      <th>Col 2 Flag</th>
      <th>Col Scalars</th>
      <th>Col 5</th>
      <th>Col 6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>100</td>
      <td>1</td>
      <td>-100</td>
      <td>20</td>
      <td>b</td>
      <td>False</td>
      <td>-5</td>
      <td>10.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>1</td>
      <td>1</td>
      <td>-9</td>
      <td>14</td>
      <td>b</td>
      <td>True</td>
      <td>5</td>
      <td>2.2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>c</th>
      <td>1</td>
      <td>1</td>
      <td>-10</td>
      <td>15</td>
      <td>c</td>
      <td>False</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>d</th>
      <td>1</td>
      <td>1</td>
      <td>-11</td>
      <td>16</td>
      <td>d</td>
      <td>False</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>e</th>
      <td>ff</td>
      <td>ff</td>
      <td>-3</td>
      <td>dd</td>
      <td>e</td>
      <td>True</td>
      <td>10</td>
      <td>12.0</td>
      <td>12.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Get dataframe value from column and index label
df['Col 1'][1] #Fetching value at (col 1, 1)
```




    1




```python
# Slice the dataframe. A dataframe can be sliced by passing the columns and row labels to slice from.
# The returned series will also contain the index of the sliced rows.
df['Col 1'][:2]
```




    a    100
    b      1
    Name: Col 1, dtype: object




```python
df['Col 1'][2]
```




    1



### Axis-wise operations
As discussed in the introduction section of dataframe, a datafrme has two axes, axis along rows is called _Axis 1_ and along columns is called _Axis 0_.

Certain operations such as droping, calculating sum, mean, etc are performed along an axis. Such operations are called axis wise operations. See figure below to get an idea of how axis wise operations are performed.  
<hr>
<center><b>An example of axis-wise operation</b></center>

<br>

![axes__description-2.svg](attachment:axes__description-2.svg)

<hr>

Several row-wise or column wise operations on a dataframe can be performed by passing the appropriate axis. 


```python
# Delete (drop) a column by passing the column label and column axis (axis 1).
# When axix=1 is passed, the operation flows along axis 1.
# This causes the operation to flow through each value along each row,
# and when it hits the value present at the location of 'Col 1', it removes it.
# This is carried out for each row (axis=1), thus effectively removing all the values
# of that perticular column.
df.drop('Col 1', axis=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Col1_copy</th>
      <th>Col 2</th>
      <th>Col 4</th>
      <th>Col A</th>
      <th>Col 2 Flag</th>
      <th>Col Scalars</th>
      <th>Col 5</th>
      <th>Col 6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1</td>
      <td>-100</td>
      <td>20</td>
      <td>b</td>
      <td>False</td>
      <td>-5</td>
      <td>10.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>1</td>
      <td>-9</td>
      <td>14</td>
      <td>b</td>
      <td>True</td>
      <td>5</td>
      <td>2.2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>c</th>
      <td>1</td>
      <td>-10</td>
      <td>15</td>
      <td>c</td>
      <td>False</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>d</th>
      <td>1</td>
      <td>-11</td>
      <td>16</td>
      <td>d</td>
      <td>False</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>e</th>
      <td>ff</td>
      <td>-3</td>
      <td>dd</td>
      <td>e</td>
      <td>True</td>
      <td>10</td>
      <td>12.0</td>
      <td>12.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Drop multiple columns
df.drop(['Col 1','Col 2'], axis=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Col1_copy</th>
      <th>Col 4</th>
      <th>Col A</th>
      <th>Col 2 Flag</th>
      <th>Col Scalars</th>
      <th>Col 5</th>
      <th>Col 6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1</td>
      <td>20</td>
      <td>b</td>
      <td>False</td>
      <td>-5</td>
      <td>10.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>1</td>
      <td>14</td>
      <td>b</td>
      <td>True</td>
      <td>5</td>
      <td>2.2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>c</th>
      <td>1</td>
      <td>15</td>
      <td>c</td>
      <td>False</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>d</th>
      <td>1</td>
      <td>16</td>
      <td>d</td>
      <td>False</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>e</th>
      <td>ff</td>
      <td>dd</td>
      <td>e</td>
      <td>True</td>
      <td>10</td>
      <td>12.0</td>
      <td>12.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Drop a row by passing the row label and row axis (axis 0).
# When axix=0 is passed, the operation flows along axis 0 (vertically).
# This causes the operation to flow through each value along each column,
# and when it hits the value present at the location of 'a', it removes it.
# This is carried out for each column (axis=0), thus effectively removing all the values
# of that perticular row.
df.drop('a', axis=0)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Col 1</th>
      <th>Col1_copy</th>
      <th>Col 2</th>
      <th>Col 4</th>
      <th>Col A</th>
      <th>Col 2 Flag</th>
      <th>Col Scalars</th>
      <th>Col 5</th>
      <th>Col 6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>b</th>
      <td>1</td>
      <td>1</td>
      <td>-9</td>
      <td>14</td>
      <td>b</td>
      <td>True</td>
      <td>5</td>
      <td>2.2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>c</th>
      <td>1</td>
      <td>1</td>
      <td>-10</td>
      <td>15</td>
      <td>c</td>
      <td>False</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>d</th>
      <td>1</td>
      <td>1</td>
      <td>-11</td>
      <td>16</td>
      <td>d</td>
      <td>False</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>e</th>
      <td>ff</td>
      <td>ff</td>
      <td>-3</td>
      <td>dd</td>
      <td>e</td>
      <td>True</td>
      <td>10</td>
      <td>12.0</td>
      <td>12.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Drop multiple rows
df.drop(['a','b'], axis=0)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Col 1</th>
      <th>Col1_copy</th>
      <th>Col 2</th>
      <th>Col 4</th>
      <th>Col A</th>
      <th>Col 2 Flag</th>
      <th>Col Scalars</th>
      <th>Col 5</th>
      <th>Col 6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>c</th>
      <td>1</td>
      <td>1</td>
      <td>-10</td>
      <td>15</td>
      <td>c</td>
      <td>False</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>d</th>
      <td>1</td>
      <td>1</td>
      <td>-11</td>
      <td>16</td>
      <td>d</td>
      <td>False</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>e</th>
      <td>ff</td>
      <td>ff</td>
      <td>-3</td>
      <td>dd</td>
      <td>e</td>
      <td>True</td>
      <td>10</td>
      <td>12.0</td>
      <td>12.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Get the sum of all values in each column.
# When axix=0 is passed, the operation flows along axis 0.
# Resulting in a summation of values in lists that are aligned vertically (columns).

# Since string values can't be mathematically operated on, any summing operations on string values will result
# in a TypeError.
# So before summing the columns, we'll drop the columns and rows containing string values.

# Drop column 'Col A' and row 'e' since they contain strings.
df.drop(columns=['Col A'],index='e', inplace=True)

# Now sum the columns
df.sum(axis=0)
```




    Col 1           103
    Col1_copy         4
    Col 2          -130
    Col 4            65
    Col 2 Flag        1
    Col Scalars      10
    Col 5          12.2
    Col 6           2.0
    dtype: object




```python
# Get the sum of all values in each row.
# When axix=1 is passed, the operation flows along axis 1.
# Resulting in a summation of values in lists that are aligned horizontally (rows).
df.sum(axis=1)
```




    a    27.0
    b    16.2
    c      12
    d      12
    dtype: object




```python
# Calculate the mean of the columns.
# Operation logic of axis same as that of df.sum() 
df.mean(axis=0)
```




    Col 1          25.75
    Col1_copy        1.0
    Col 2          -32.5
    Col 4          16.25
    Col 2 Flag      0.25
    Col Scalars      2.5
    Col 5            6.1
    Col 6            1.0
    dtype: object




```python
# Calculate the mean of the rows.
df.mean(axis=1)
```




    a    3.375
    b    2.025
    c      2.0
    d      2.0
    dtype: object



### Operating on a Dataframe


```python
# Combining multiple dataframes.
# When combining multiple dataframes, a dataframe with a union of the rows and columns of the original dataframes is returned.
df + df_new
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Col 1</th>
      <th>Col 2</th>
      <th>Col 2 Flag</th>
      <th>Col 4</th>
      <th>Col 5</th>
      <th>Col 6</th>
      <th>Col A</th>
      <th>Col Scalars</th>
      <th>Col1_copy</th>
      <th>New_Col</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>101</td>
      <td>-108</td>
      <td>True</td>
      <td>33</td>
      <td>11.1</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>b</th>
      <td>2</td>
      <td>-18</td>
      <td>True</td>
      <td>28</td>
      <td>4.4</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>10</td>
      <td>2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>c</th>
      <td>2</td>
      <td>-20</td>
      <td>False</td>
      <td>30</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10</td>
      <td>2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>d</th>
      <td>2</td>
      <td>-22</td>
      <td>False</td>
      <td>32</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10</td>
      <td>2</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Element wise operations
df * 2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Col 1</th>
      <th>Col1_copy</th>
      <th>Col 2</th>
      <th>Col 4</th>
      <th>Col 2 Flag</th>
      <th>Col Scalars</th>
      <th>Col 5</th>
      <th>Col 6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>200</td>
      <td>2</td>
      <td>-200</td>
      <td>40</td>
      <td>0</td>
      <td>-10</td>
      <td>20.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>2</td>
      <td>2</td>
      <td>-18</td>
      <td>28</td>
      <td>2</td>
      <td>10</td>
      <td>4.4</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>c</th>
      <td>2</td>
      <td>2</td>
      <td>-20</td>
      <td>30</td>
      <td>0</td>
      <td>10</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>d</th>
      <td>2</td>
      <td>2</td>
      <td>-22</td>
      <td>32</td>
      <td>0</td>
      <td>10</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Transpose the dataframe. Transposion is the process of converting the rows to columns and vise versa.
df.T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Col 1</th>
      <td>100</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Col1_copy</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Col 2</th>
      <td>-100</td>
      <td>-9</td>
      <td>-10</td>
      <td>-11</td>
    </tr>
    <tr>
      <th>Col 4</th>
      <td>20</td>
      <td>14</td>
      <td>15</td>
      <td>16</td>
    </tr>
    <tr>
      <th>Col 2 Flag</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>Col Scalars</th>
      <td>-5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Col 5</th>
      <td>10.0</td>
      <td>2.2</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Col 6</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Most numpy functions can be called directly on series and dataframe.
np.multiply(df,3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Col 1</th>
      <th>Col1_copy</th>
      <th>Col 2</th>
      <th>Col 4</th>
      <th>Col 2 Flag</th>
      <th>Col Scalars</th>
      <th>Col 5</th>
      <th>Col 6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>300</td>
      <td>3</td>
      <td>-300</td>
      <td>60</td>
      <td>0</td>
      <td>-15</td>
      <td>30.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>3</td>
      <td>3</td>
      <td>-27</td>
      <td>42</td>
      <td>3</td>
      <td>15</td>
      <td>6.6</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>c</th>
      <td>3</td>
      <td>3</td>
      <td>-30</td>
      <td>45</td>
      <td>0</td>
      <td>15</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>d</th>
      <td>3</td>
      <td>3</td>
      <td>-33</td>
      <td>48</td>
      <td>0</td>
      <td>15</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
np.exp(df['Col 2 Flag'])
```




    a    1.00000
    b    2.71875
    c    1.00000
    d    1.00000
    Name: Col 2 Flag, dtype: float16




```python
np.log10(df['Col Scalars'])
```

    /home/somanna/envs/mainenv/lib/python3.11/site-packages/pandas/core/arraylike.py:396: RuntimeWarning: invalid value encountered in log10
      result = getattr(ufunc, method)(*inputs, **kwargs)
    




    a        NaN
    b    0.69897
    c    0.69897
    d    0.69897
    Name: Col Scalars, dtype: float64



### Ways to summerize a Dataframe


```python
# Get the first few entries (rows) of the dataframe.
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Col 1</th>
      <th>Col1_copy</th>
      <th>Col 2</th>
      <th>Col 4</th>
      <th>Col 2 Flag</th>
      <th>Col Scalars</th>
      <th>Col 5</th>
      <th>Col 6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>100</td>
      <td>1</td>
      <td>-100</td>
      <td>20</td>
      <td>False</td>
      <td>-5</td>
      <td>10.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>1</td>
      <td>1</td>
      <td>-9</td>
      <td>14</td>
      <td>True</td>
      <td>5</td>
      <td>2.2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>c</th>
      <td>1</td>
      <td>1</td>
      <td>-10</td>
      <td>15</td>
      <td>False</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>d</th>
      <td>1</td>
      <td>1</td>
      <td>-11</td>
      <td>16</td>
      <td>False</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# The head() method by default shows the first five entries,
# you can choose the number of entries to show by passing
# the number of entries to the head method.

# Show first ten entries.
df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Col 1</th>
      <th>Col1_copy</th>
      <th>Col 2</th>
      <th>Col 4</th>
      <th>Col 2 Flag</th>
      <th>Col Scalars</th>
      <th>Col 5</th>
      <th>Col 6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>100</td>
      <td>1</td>
      <td>-100</td>
      <td>20</td>
      <td>False</td>
      <td>-5</td>
      <td>10.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>1</td>
      <td>1</td>
      <td>-9</td>
      <td>14</td>
      <td>True</td>
      <td>5</td>
      <td>2.2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>c</th>
      <td>1</td>
      <td>1</td>
      <td>-10</td>
      <td>15</td>
      <td>False</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>d</th>
      <td>1</td>
      <td>1</td>
      <td>-11</td>
      <td>16</td>
      <td>False</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Get the last few entries of the dataframe.
df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Col 1</th>
      <th>Col1_copy</th>
      <th>Col 2</th>
      <th>Col 4</th>
      <th>Col 2 Flag</th>
      <th>Col Scalars</th>
      <th>Col 5</th>
      <th>Col 6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>100</td>
      <td>1</td>
      <td>-100</td>
      <td>20</td>
      <td>False</td>
      <td>-5</td>
      <td>10.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>1</td>
      <td>1</td>
      <td>-9</td>
      <td>14</td>
      <td>True</td>
      <td>5</td>
      <td>2.2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>c</th>
      <td>1</td>
      <td>1</td>
      <td>-10</td>
      <td>15</td>
      <td>False</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>d</th>
      <td>1</td>
      <td>1</td>
      <td>-11</td>
      <td>16</td>
      <td>False</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Get the last 3 entries
df.tail(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Col 1</th>
      <th>Col1_copy</th>
      <th>Col 2</th>
      <th>Col 4</th>
      <th>Col 2 Flag</th>
      <th>Col Scalars</th>
      <th>Col 5</th>
      <th>Col 6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>b</th>
      <td>1</td>
      <td>1</td>
      <td>-9</td>
      <td>14</td>
      <td>True</td>
      <td>5</td>
      <td>2.2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>c</th>
      <td>1</td>
      <td>1</td>
      <td>-10</td>
      <td>15</td>
      <td>False</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>d</th>
      <td>1</td>
      <td>1</td>
      <td>-11</td>
      <td>16</td>
      <td>False</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Get information about the datframe.
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 4 entries, a to d
    Data columns (total 8 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   Col 1        4 non-null      object 
     1   Col1_copy    4 non-null      object 
     2   Col 2        4 non-null      int64  
     3   Col 4        4 non-null      object 
     4   Col 2 Flag   4 non-null      bool   
     5   Col Scalars  4 non-null      int64  
     6   Col 5        2 non-null      float64
     7   Col 6        2 non-null      float64
    dtypes: bool(1), float64(2), int64(2), object(3)
    memory usage: 432.0+ bytes
    


```python
# Show descriptive statistics of the columns.
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Col 2</th>
      <th>Col Scalars</th>
      <th>Col 5</th>
      <th>Col 6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4.000000</td>
      <td>4.0</td>
      <td>2.000000</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-32.500000</td>
      <td>2.5</td>
      <td>6.100000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>45.007407</td>
      <td>5.0</td>
      <td>5.515433</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-100.000000</td>
      <td>-5.0</td>
      <td>2.200000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-33.250000</td>
      <td>2.5</td>
      <td>4.150000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-10.500000</td>
      <td>5.0</td>
      <td>6.100000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>-9.750000</td>
      <td>5.0</td>
      <td>8.050000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>-9.000000</td>
      <td>5.0</td>
      <td>10.000000</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# The describe() method by default shows statistics of numerical columns only
# as shown in the output above.
# To show statistics for all columns, pass 'include='all''
df.describe(include='all')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Col 1</th>
      <th>Col1_copy</th>
      <th>Col 2</th>
      <th>Col 4</th>
      <th>Col 2 Flag</th>
      <th>Col Scalars</th>
      <th>Col 5</th>
      <th>Col 6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.000000</td>
      <td>4.0</td>
      <td>4</td>
      <td>4.0</td>
      <td>2.000000</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>20.0</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>3.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>-32.500000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.5</td>
      <td>6.100000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>45.007407</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>5.515433</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>-100.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-5.0</td>
      <td>2.200000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>-33.250000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.5</td>
      <td>4.150000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>-10.500000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>6.100000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>-9.750000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>8.050000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>-9.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>10.000000</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



### Load a Dataset from a Stream
You can load data from other data structures such as a csv or json file into a dataframe from various streams such as a local disk or a network location.


```python
# Load data from a csv file
df = pd.read_csv('real_estate.csv')
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>status</th>
      <th>bed</th>
      <th>bath</th>
      <th>acre_lot</th>
      <th>city</th>
      <th>state</th>
      <th>zip_code</th>
      <th>house_size</th>
      <th>prev_sold_date</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>for_sale</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>0.12</td>
      <td>Adjuntas</td>
      <td>Puerto Rico</td>
      <td>601.0</td>
      <td>920.0</td>
      <td>NaN</td>
      <td>105000.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>for_sale</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>0.08</td>
      <td>Adjuntas</td>
      <td>Puerto Rico</td>
      <td>601.0</td>
      <td>1527.0</td>
      <td>NaN</td>
      <td>80000.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>for_sale</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.15</td>
      <td>Juana Diaz</td>
      <td>Puerto Rico</td>
      <td>795.0</td>
      <td>748.0</td>
      <td>NaN</td>
      <td>67000.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>for_sale</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>0.10</td>
      <td>Ponce</td>
      <td>Puerto Rico</td>
      <td>731.0</td>
      <td>1800.0</td>
      <td>NaN</td>
      <td>145000.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>for_sale</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>0.05</td>
      <td>Mayaguez</td>
      <td>Puerto Rico</td>
      <td>680.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>65000.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>99995</th>
      <td>for_sale</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>Hudson</td>
      <td>Massachusetts</td>
      <td>1749.0</td>
      <td>2864.0</td>
      <td>NaN</td>
      <td>749900.0</td>
    </tr>
    <tr>
      <th>99996</th>
      <td>for_sale</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.34</td>
      <td>Auburn</td>
      <td>Massachusetts</td>
      <td>1501.0</td>
      <td>1075.0</td>
      <td>1999-06-07</td>
      <td>349900.0</td>
    </tr>
    <tr>
      <th>99997</th>
      <td>for_sale</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>1.01</td>
      <td>Shrewsbury</td>
      <td>Massachusetts</td>
      <td>1545.0</td>
      <td>1632.0</td>
      <td>1995-09-27</td>
      <td>549000.0</td>
    </tr>
    <tr>
      <th>99998</th>
      <td>for_sale</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>0.12</td>
      <td>Worcester</td>
      <td>Massachusetts</td>
      <td>1604.0</td>
      <td>1332.0</td>
      <td>2000-09-11</td>
      <td>299000.0</td>
    </tr>
    <tr>
      <th>99999</th>
      <td>for_sale</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>21.67</td>
      <td>Grafton</td>
      <td>Massachusetts</td>
      <td>1536.0</td>
      <td>1846.0</td>
      <td>2020-10-06</td>
      <td>535000.0</td>
    </tr>
  </tbody>
</table>
<p>100000 rows × 10 columns</p>
</div>



### Plot the Data in a Dataframe


```python
df.price.plot()
```




    <Axes: >




    
![png](output_164_1.png)
    



```python
df.state.hist()
```




    <Axes: >




    
![png](output_165_1.png)
    



```python
# Pandas uses Matplotlib to plot data. Hence you can use matplotlib to modify the plots from Pandas.

import matplotlib.pyplot as plt

plt.figure(figsize=(20,5))
df.state.hist()
```




    <Axes: >




    
![png](output_166_1.png)
    


You may also explore other types of plots Pandas can generate.
### Obtain and Manipulate Null Values


```python
# Show what values are null in the dataframe
df.isna()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>status</th>
      <th>bed</th>
      <th>bath</th>
      <th>acre_lot</th>
      <th>city</th>
      <th>state</th>
      <th>zip_code</th>
      <th>house_size</th>
      <th>prev_sold_date</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>99995</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>99996</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>99997</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>99998</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>99999</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>100000 rows × 10 columns</p>
</div>



Null values are masked as True, non-null values are masked as False.


```python
# Get the count of null values in each column
df.isna().sum()
```




    status                0
    bed               24950
    bath              24888
    acre_lot          14013
    city                 52
    state                 0
    zip_code            195
    house_size        24918
    prev_sold_date    71255
    price                 0
    dtype: int64



By default, the sum() method is applied over columns (axis=0) and thus gets the sum of values of each column, to get the sum of values of each row instead, pass 'axis=1' to the sum() method.


```python
df.isna().sum(axis=1)
```




    0        1
    1        1
    2        1
    3        1
    4        2
            ..
    99995    2
    99996    0
    99997    0
    99998    0
    99999    0
    Length: 100000, dtype: int64



<center><b><font size='6'>--X-- END --X--</font></b></center>
