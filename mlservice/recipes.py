def recipe_quantize(input, label, output, num_splits, train_sw_colname, spark=False):
    import mlservice.stages.sklearn_stages as mls_basic

    if not spark:
        return mls_basic.Pipeline(stages=
                                  [mls_basic.Quantize(input=input, label=label, output=output,
                                                      train_sw_colname=train_sw_colname,
                                                      len=num_splits),
                                   mls_basic.Compressor(input=output, output=output)])
    else:
        import mlservice.stages.spark_stages as mls_spark

        return mls_spark.Pipeline(stages=
                                  [mls_spark.Quantize(input=input, label=label, output=output,
                                                      train_sw_colname=train_sw_colname,
                                                      len=num_splits),
                                   # mls_spark.Compressor(input=output, output=output)
                                   ])


def recipe_quantize_1hot(input, label, output, num_splits, train_sw_colname, spark=False):
    import mlservice.stages.sklearn_stages as mls_basic

    if not spark:
        return mls_basic.Pipeline(stages=
                                  [mls_basic.Quantize(input=input, label=label, output=output,
                                                      train_sw_colname=train_sw_colname,
                                                      len=num_splits),
                                   mls_basic.Compressor(input=output, output=output),
                                   mls_basic.Onehot(input=output, output=output)])
    else:
        import mlservice.stages.spark_stages as mls_spark

        return mls_spark.Pipeline(stages=
                                  [mls_spark.Quantize(input=input, label=label, output=output + '_q',
                                                      train_sw_colname=train_sw_colname,
                                                      len=num_splits),
                                   # mls_spark.Compressor(input=output, output=output),
                                   mls_spark.Onehot(input=output + '_q', output=output)])
