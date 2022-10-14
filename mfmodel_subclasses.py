## There is a better way to do the number of units in the layers, but it may only work when you exclusively use the def build_model(hp) method of HP tuning. 
## f"units{i}"
class hyperMLP(kt.HyperModel):
    def __init__(
        self,regType,regValue, hiddenLayerActivation, 
        outputLayerActivation,kernelInitializer, 
        optimizer, loss, input_data,output_data
        ):

        self.regType = regType
        self.regValue = regValue
        self.hiddenLayerActivation = hiddenLayerActivation
        self.outputLayerActivation = outputLayerActivation
        self.kernelInitializer = kernelInitializer
        self.optimizer = optimizer
        self.loss = loss
        self.input_data = input_data
        self.output_data = output_data 

    def build(self,hp):
        inputlayershape = int(len(self.input_data[0,:]))
        outputlayershape = int(len(self.output_data[0,:]))

        hp_units1 = hp.Int('units1', min_value=32, max_value=80, step=8)
        hp_units2 = hp.Int('units2', min_value=32, max_value=80, step=8)
        hp_units3 = hp.Int('units3', min_value=32, max_value=80, step=8)
        hp_layers = hp.Int('layers', min_value=1, max_value =3,step=1)
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        hp_units_list = [hp_units1, hp_units2, hp_units3]
        
        model = keras.Sequential()
        model.add(tf.keras.layers.Dense(inputlayershape))
        for layerNumber in np.arange(hp_layers):
            model.add(tf.keras.layers.Dense(
                hp_units_list[layerNumber],
                activation = self.outputLayerActivation,
                kernel_regularizer = self.regType(self.regValue),
                kernel_initializer = self.kernelInitializer
        ))
        model.add(tf.keras.layers.Dense(outputlayershape, dtype='float32'))
        model.compile(
            optimizer = self.optimizer(learning_rate=hp_learning_rate),
            loss = self.loss,
            metrics = [tf.keras.metrics.MeanSquaredError()],
            steps_per_execution=10
                    )
        return model

class hyperMLPv2(kt.HyperModel):
    def __init__(
        self,regType,regValue, hiddenLayerActivation, 
        outputLayerActivation,kernelInitializer, 
        optimizer, loss, input_data,output_data
        ):

        self.regType = regType
        self.regValue = regValue
        self.hiddenLayerActivation = hiddenLayerActivation
        self.outputLayerActivation = outputLayerActivation
        self.kernelInitializer = kernelInitializer
        self.optimizer = optimizer
        self.loss = loss
        self.input_data = input_data
        self.output_data = output_data 

    def build(self,hp):
        inputlayershape = int(len(self.input_data[0,:]))
        outputlayershape = int(len(self.output_data[0,:]))

        hp_units1 = hp.Int('units1', min_value=32, max_value=80, step=8)
        hp_units2 = hp.Int('units2', min_value=32, max_value=80, step=8)
        hp_units3 = hp.Int('units3', min_value=32, max_value=80, step=8)
        hp_layers = hp.Int('layers', min_value=1, max_value =3,step=1)
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        hp_units_list = [hp_units1, hp_units2, hp_units3]
        
        model = keras.Sequential()
        model.add(tf.keras.layers.Dense(inputlayershape))

        for layerNumber in np.arange(hp_layers):
            model.add(tf.keras.layers.Dense(
                units = hp.Int(f"units{layerNumber}", min_value=32,max_value=80, steps=8),
                activation = self.outputLayerActivation,
                kernel_regularizer = self.regType(self.regValue),
                kernel_initializer = self.kernelInitializer
        ))
        model.add(tf.keras.layers.Dense(outputlayershape, dtype='float32'))
        model.compile(
            optimizer = self.optimizer(learning_rate=hp_learning_rate),
            loss = self.loss,
            metrics = [tf.keras.metrics.MeanSquaredError()],
            steps_per_execution=10
                    )
        return model

class MemoryPrintingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
      gpu_dict = tf.config.experimental.get_memory_info('GPU:0')
      tf.print('\n GPU memory details [current: {} gb, peak: {} gb]'.format(
          float(gpu_dict['current']) / (1024 ** 3), 
          float(gpu_dict['peak']) / (1024 ** 3)))

class MemorySavingCallback(tf.keras.callbacks.Callback):
    def __init__(self,memorySaveListName):
        self.memorySaveListName = memorySaveListName
    def on_epoch_end(self, epoch, logs=None):
        gpu_dict = tf.config.experimental.get_memory_info('GPU:0')
        #this is hackish. sorry for anyone who has the misfortune to see how I'm doing this. 
        globals()[self.memorySaveListName].append(float(gpu_dict['peak']) / (1024 ** 3))