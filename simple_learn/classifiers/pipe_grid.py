from sklearn.preprocessing import StandardScaler, QuantileTransformer, MaxAbsScaler, MinMaxScaler

# Hyper-parameter search space for grid search
pipe_process_map = {
    'scaler': ('scaler', StandardScaler()),
    'quant': ('quant', QuantileTransformer()),
    'maxabs': ('maxabs', MaxAbsScaler()),
    'minmax': ('minmax', MinMaxScaler())
}