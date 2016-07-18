'''
Module to construct full feature grid (as a dataframe or 2D array)

Will handle various attempts at dimensionality reduction that we come up with by hand.
'''

import os
import glob
import pickle
import itertools
import sys
import pandas as pd
import numpy as np

from IPython import embed

def gps_transform(func):
    '''
    Decorator to mark function for use in gps dim reduction.

    These functions take a dataframe of gps data matched to a task
    and return a _list_ of _pd.Series_ which are the features for each generated row
    '''
    def feature_func(df, **kwargs):
        # These are all the same so max just gets a single value
        default_features = pd.Series(dict(label=df.task_label.max(), name=df.name.max(), start_time=df.start_time.max(), end_time=df.end_time.max())) 
        rows = []
        for row in func(df, **kwargs):
            rows.append(pd.concat((default_features, row)))
        return rows
    gps_transform.funcs[func.__name__] = feature_func
    return feature_func

gps_transform.funcs = {}


def cached(func):
    '''
    Decorator to optionally cache a DataFrame to file .

    This decoration relies on func taking only simple (hashable) kwargs
    Using pickle for now. If performance gets slow consider msgpack or hdf5
    '''
    def cached_func(*args, **kwargs):
        c = kwargs.pop('cache', False)
        context = "data/{}.pkl".format(hash(frozenset(kwargs.items())))
        if c:
            # Attempt to retrieve from file
            if os.path.exists(context):
                return pd.read_pickle(context)
        df = func(*args, **kwargs)
        if c:
            # Load to file if cache specified
            df.to_pickle(context)
        return df
    return cached_func


@gps_transform
def padded(df):
    '''
    Retain all of the x,y,z as features, padded into an 8 hour window.
    '''
    # Get times as seconds since start and use as index
    df.index = ((df.position_update_timestamp - df.position_update_timestamp.min())/pd.Timedelta('1s')).astype('int') 
    # Drop everything but x,y,z
    df = df['position_x','position_y','position_z']

    # Reindex to pad out to a length of 8h
    df.reindex(np.arange(8*3600, fill_value=0))

    # Now build a feature vector from it. It'll be big :(
    return [pd.concat(df.position_x, df.position_y, df.position_z)]
    

def distance(*args):
    '''
    N-dim euclidean distance
    '''
    return np.sqrt(np.sum(a*a for a in args))

@gps_transform
def chunked(df, **kwargs):
    '''
    Extract a few statistical values from time data chunked over an interval (1m).
    Keep an hour's worth.
    '''
    # First calcluate any new values
    # Velocity is distance between consecutive points per sec
    df['velocity'] = distance(df.position_x.diff(), df.position_y.diff())/(df.position_update_timestamp.diff()/pd.Timedelta('1s'))
    df['velocity_x'] = distance(df.position_x.diff())/(df.position_update_timestamp.diff()/pd.Timedelta('1s'))
    df['velocity_y'] = distance(df.position_y.diff())/(df.position_update_timestamp.diff()/pd.Timedelta('1s'))
    df['acceleration'] = distance(df.velocity_x.diff(), df.velocity_y.diff())/(df.position_update_timestamp.diff()/pd.Timedelta('1s'))
    # List of columns to form features
    # cols = ['position_x','position_y','position_z','velocity']
    cols = ['position_x', 'position_y','velocity', 'velocity_x', 'velocity_y', 'acceleration']
    # Apparently sometimes the gps data is not consecutive in seconds
    # so we need to focus on timestamps and not indices
    interval = pd.Timedelta(kwargs.pop('interval','10m')) # Length of interval for each output row
    sub_interval = pd.Timedelta(kwargs.pop('subinterval','2m')) # Sub interval in which to sample derived quantities
    dens = float(kwargs.pop('dens','1.0'))
    n_sub = int(interval/(dens*sub_interval))

    rows = []

    # Create a chunk containing all timestamps within one interval
    lower = df.position_update_timestamp.min() 
    upper = lower + interval
    chunk = df[(df.position_update_timestamp > lower) & (df.position_update_timestamp < upper)].copy()

    while len(chunk):
        # Calculate values in the sub intervals for this chunk.
        means = []
        moveon=False
        for col in cols:
            mean = chunk[col].groupby(((chunk.position_update_timestamp - chunk.position_update_timestamp.min())/sub_interval).astype(int)).mean()
            if (len(mean) < n_sub) or (mean.var()==0.0):
                moveon=True
            mean = mean.reindex(range(int(interval/sub_interval)), method='nearest')
            means.append(mean)
        stds = []
        # print moveon
        for col in cols:
            std = chunk[col].groupby(((chunk.position_update_timestamp - chunk.position_update_timestamp.min())/sub_interval).astype(int)).std()
            std = std.reindex(range(int(interval/sub_interval)), method='nearest')
            stds.append(std)
        features = pd.concat((pd.concat(means), pd.concat(stds)))
        features.index = range(len(features))
        # Get the next chunk
        lower,upper = lower+interval, upper+interval
        chunk = df[(df.position_update_timestamp > lower) & (df.position_update_timestamp < upper)].copy()
        if not moveon: rows.append(features)
    return rows


@gps_transform
def consecutives():
    '''
    Map gps to a list of consecutive stationary points (x,y,z,t,v)
    where t is total time at that location and v is velocity to next
    '''
    raise KeyError("consecutives not yet implemented")
    
def create_gps_pickles():
    '''
    Create a series of files that store the subset of gps data for each distinct task.
    '''
    
    # First we load the timestamps DF (just 100 for testing)
    # df = pd.read_pickle('data/TaskCodeTimestamps.pkl')[:100]
    df = pd.read_pickle('data/TaskCodeTimestamps.pkl')
    df['duration'] = (df.end_time - df.start_time) / pd.Timedelta('1h')    
    df = df[df.duration <= 8]
    sizes = df.groupby(df.task).size()
    common = sizes[sizes > 10].index
    df = df[df.task.isin(common)]
    
    # Grab gps data for each task, processing if specified
    gps = pd.read_pickle('data/LocationData.pkl')
    with open('data/NameToNode.pkl','r') as infile:
        nodes = pickle.load(infile)
    nodes = dict((int(key), val) for key,val in nodes.iteritems())

    df['name'] = df.first_name + ' ' + df.last_name
    gps['name'] = gps.node_id.map(nodes)

    # Build an extension to the df by creating feature vectors from (transformed) x,y,z data
    for index, name, start, end, task in itertools.izip(df.index, df.name, df.start_time, df.end_time, df.task):
        sys.stdout.write('Processing task number {0} out of {1}\r'.format(index,df.index[-1]))
        sys.stdout.flush()
        sub_gps = gps[(gps.name == name) & (gps.position_update_timestamp > start) & (gps.position_update_timestamp < end)].copy()
        if not len(sub_gps):
            continue
        sub_gps['start_time'] = start
        sub_gps['end_time'] = end
        sub_gps['task_id'] = index
        sub_gps['task_label'] = task
        sub_gps.to_pickle('data/gps_{:06d}.pkl'.format(index))
    print


@cached
def load_tasks(gps_reduce='chunked', accel_reduce=None, interval='60m', subinterval='1m', dens='1.0', n=None):
    '''
    Return a pandas data frame that stores the features and labels for each task.

    For now only gps data is supported.

    Kwargs:
    gps_reduce -- name of the transform of gps three vectors into reduced vectors (any number of outputs)!
                  Accepts the name of any function marked @gps
    '''

    # First determine the indices from the pickle files
    fnames = glob.glob('data/gps_*.pkl')
    if n is not None:
        fnames = fnames[:n]
    
    # Because each of the input files can generate multiple rows depending on 
    # the choice of transform, store all as a list first
    reduced_gps = list()
    for i,fname in enumerate(fnames):
        sys.stdout.write('Processing file number {0} out of {1}\r'.format(i,len(fnames)))
        sys.stdout.flush()
        reduced_gps.append(gps_transform.funcs[gps_reduce](pd.read_pickle(fname), interval=interval, subinterval=subinterval, dens=dens))
    rows = list(itertools.chain.from_iterable(reduced_gps))
    df = pd.DataFrame(index=range(len(rows)), columns=rows[0].index)
    for i,row in enumerate(rows):
        df.iloc[i] = row
    return df
    
    
def main():
    '''
    Make the gps pickles which are used by other methods.
    '''
    #create_gps_pickles()
    df=load_tasks(cache=True)
    
if __name__ == "__main__":
    main()
