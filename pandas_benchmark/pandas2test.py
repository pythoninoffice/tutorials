import time
import pandas as pd

pd.__version__

pd.options.mode.dtype_backend = 'pyarrow'

file_1 = '100_million_1.csv'
file_2 = '100_million_2.csv'
num_ran = 1

lib_name = 'pandas'
stats = []

for i in range (num_ran):
    stats_inner = {}
    start = time.time()
    df_1 = pd.read_csv(file_1,  engine="pyarrow", use_nullable_dtypes=True, index_col = 0)
    print(f'loading csv with {lib_name} took: {time.time() - start} seconds')
    stats_inner['loading_1'] = (time.time() - start)
    
    start = time.time()
    df_2 = pd.read_csv(file_2, engine="pyarrow", use_nullable_dtypes=True, index_col = 0)
    print(f'loading csv with {lib_name} took: {time.time() - start} seconds')
    stats_inner['loading_2'] = (time.time() - start)
    
    start = time.time()
    df_1.merge(df_2, on = 'unique_id_text' )
    print(f'merge csv on unique_id_text with {lib_name} took: {time.time() - start} seconds')
    stats_inner['merging'] = (time.time() - start)

    ## out of memory
    start=time.time()
    pd.concat([df_1, df_2])
    print(f'concat csv with {lib_name} took: {time.time() - start} seconds')
    stats_inner['concat'] = (time.time() - start)
    
##    start=time.time()
##    df_1.groupby('text_id').sum(['0','1','2','3','4','5','6'])
##    print(f'groupby and sum by text_id with {lib_name} took: {time.time() - start} seconds ')
##    stats_inner['groupby'] = (time.time() - start)
    
    start=time.time()
    df_1['new_col'] = df_1['0'].apply(round)
    print(f'apply round method with {lib_name} took {time.time() - start} seconds.')
    stats_inner['apply'] = (time.time() - start)
    
    start=time.time()
    df_1.loc[df_1['text_id'] == '9999_ID']
    print(f'.loc filter took {time.time() - start} seconds.')
    stats_inner['filtering'] = (time.time() - start)
    
    start=time.time()
    df_1.loc[df_1['text_id'] == '9999_ID', 'text_id'] = 'found it'
    print(f'.loc filter and update value took {time.time() - start} seconds.')
    stats_inner['filtering & updating'] = (time.time() - start)
    
    start=time.time()
    df_1['0_new'] = df_1['0'] * 2 + 1
    print(f'simple calculation took {time.time() - start} seconds.')
    stats_inner['column calculation'] = (time.time() - start)
    
    stats.append(stats_inner)
    
print([k for k in sorted(stats[0].keys())])
pandas_stats = [round((stats[0][k] + stats[1][k])/1,5) for k in sorted(stats[0].keys())]
print(pandas_stats)
