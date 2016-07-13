'''
Module to construct full feature grid (as a dataframe or 2D array)

Will handle various attempts at dimensionality reduction that we come up with by hand.
'''

import os
import pickle
import itertools

import pandas as pd
import numpy as np

def gps_transform(func):
    '''
    Decorator to mark function for use in gps dim reduction.

    These functions take a dataframe of gps data matched to a task
    and return a series which are the features for that task
    '''
    gps_transform.funcs[func.__name__] = func
    return func

gps_transform.funcs = {}


def hashed(d):
    return tuple(sorted((a,tuple(b)) if is_listy(b) else (a,b) for a,b in d.iteritems()))


def cached(func):
    '''
    Decorator to optionally cache a DataFrame to file .

    This decoration relies on func taking only simple (hashable) kwargs
    Using pickle for now. If performance gets slow consider msgpack or hdf5
    '''
    def cached_func(*args, **kwargs):
        c = kwargs.pop('cache', False)
        context = hash(frozenset(kwargs.items())) 
        if c:
            # Attempt to retrieve from file
            if os.path.exists(context + '.pkl'):
                return pd.read_pickle(context + '.pkl')
        df = func(*args, **kwargs)
        if c:
            # Load to file if cache specified
            df.to_pickle(context + '.pkl')
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
    return pd.concat(df.position_x, df.position_y, df.position_z)
    

def distance(*args):
    '''
    N-dim euclidean distance
    '''
    return np.sqrt(sum(a*a for a in args))

@gps_transform
def chunked(df):
    '''
    Extract a few statistical values from time data chunked over an interval (1m).
    Keep an hour's worth.
    '''
    # Clean up index to be just integers
    # Keep only first on hour and pad out to one hour if necessary
    df.index = range(len(df.index))
    df = df.reindex(xrange(3600))

    # Velocity is distance between consecutive points (/s)
    df['velocity'] = distance(df.position_x.diff(), df.position_y.diff(), df.position_z.diff())
    
    # Get mean, std for now
    mean = pd.concat(df[col].groupby(df.index/60).mean() for col in ['position_x', 'position_y', 'position_z','velocity'])
    std  = pd.concat(df[col].groupby(df.index/60).std()  for col in ['position_x', 'position_y', 'position_z','velocity'])
    return pd.concat((mean, std))


@gps_transform
def consecutives():
    '''
    Map gps to a list of consecutive stationary points (x,y,z,t,v)
    where t is total time at that location and v is velocity to next
    '''
    raise KeyError("consecutives not yet implemented")
    

@cached
def load_tasks(gps_reduce='chunked', accel_reduce=None, interval=None):
    '''
    Return a pandas data frame that stores the features and labels for each task.

    For now only gps data is supported.

    Kwargs:
    gps_reduce -- name of the transform of gps three vectors into reduced vectors (any number of outputs)!
                  Accepts the name of any function marked @gps
    '''
    
    # First we load the timestamps DF - it will be extended into the whole dataset
    df = pd.read_pickle('TaskCodeTimestamps.pkl')[:100]

    # Grab gps data for each task, processing if specified
    gps = pd.read_pickle('LocationData.pkl')
    with open('NameToNode.pkl','r') as infile:
        nodes = pickle.load(infile)
    nodes = dict((int(key), val) for key,val in nodes.iteritems())

    df['name'] = df.first_name + ' ' + df.last_name
    gps['name'] = gps.node_id.map(nodes)

    # Build an extension to the df by creating feature vectors from (transformed) x,y,z data
    extension = None
    for index, name, start, end in itertools.izip(df.index, df.name, df.start_time, df.end_time):
        sub_gps = gps[(gps.name == name) & (gps.position_update_timestamp > start) & (gps.position_update_timestamp < end)]
        if not len(sub_gps):
            continue
        features = gps_transform.funcs[gps_reduce](sub_gps)
        if extension is None:
            extension = pd.DataFrame(columns=features.index, index=gps.index)
        extension.loc[index] = features
    return pd.concat((df, extension), axis=1)
    
    
def main():
    '''
    Quick tests
    '''
    df = load_tasks()
    print df.head()


if __name__ == "__main__":
    main()
