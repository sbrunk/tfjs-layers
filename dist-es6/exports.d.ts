import { io, Tensor } from '@tensorflow/tfjs-core';
import { Constraint, MaxNormConfig, MinMaxNormConfig, UnitNormConfig } from './constraints';
import { ContainerConfig, InputConfig, InputLayerConfig, Layer, LayerConfig } from './engine/topology';
import { Model } from './engine/training';
import { ConstantConfig, IdentityConfig, Initializer, OrthogonalConfig, RandomNormalConfig, RandomUniformConfig, SeedOnlyInitializerConfig, TruncatedNormalConfig, VarianceScalingConfig, Zeros } from './initializers';
import { ELULayerConfig, LeakyReLULayerConfig, SoftmaxLayerConfig, ThresholdedReLULayerConfig } from './layers/advanced_activations';
import { ConvLayerConfig, Cropping2DLayerConfig, SeparableConvLayerConfig } from './layers/convolutional';
import { DepthwiseConv2DLayerConfig } from './layers/convolutional_depthwise';
import { ActivationLayerConfig, DenseLayerConfig, DropoutLayerConfig, RepeatVectorLayerConfig, ReshapeLayerConfig } from './layers/core';
import { EmbeddingLayerConfig } from './layers/embeddings';
import { ConcatenateLayerConfig } from './layers/merge';
import { BatchNormalizationLayerConfig } from './layers/normalization';
import { ZeroPadding2DLayerConfig } from './layers/padding';
import { GlobalPooling2DLayerConfig, Pooling1DLayerConfig, Pooling2DLayerConfig } from './layers/pooling';
import { GRUCellLayerConfig, GRULayerConfig, LSTMCellLayerConfig, LSTMLayerConfig, RNN, RNNCell, RNNLayerConfig, SimpleRNNCellLayerConfig, SimpleRNNLayerConfig, StackedRNNCellsConfig } from './layers/recurrent';
import { BidirectionalLayerConfig, Wrapper, WrapperLayerConfig } from './layers/wrappers';
import { Sequential, SequentialConfig } from './models';
import { L1Config, L1L2Config, L2Config, Regularizer } from './regularizers';
import { SymbolicTensor } from './types';
export declare class ModelExports {
    static model(config: ContainerConfig): Model;
    static sequential(config?: SequentialConfig): Sequential;
    static loadModel(pathOrIOHandler: string | io.IOHandler): Promise<Model>;
    static input(config: InputConfig): SymbolicTensor;
}
export declare class LayerExports {
    static Layer: typeof Layer;
    static RNN: typeof RNN;
    static RNNCell: typeof RNNCell;
    static inputLayer(config: InputLayerConfig): Layer;
    static input: typeof ModelExports.input;
    static elu(config?: ELULayerConfig): Layer;
    static leakyReLU(config?: LeakyReLULayerConfig): Layer;
    static softmax(config?: SoftmaxLayerConfig): Layer;
    static thresholdedReLU(config?: ThresholdedReLULayerConfig): Layer;
    static conv1d(config: ConvLayerConfig): Layer;
    static conv2d(config: ConvLayerConfig): Layer;
    static conv2dTranspose(config: ConvLayerConfig): Layer;
    static separableConv2d(config: SeparableConvLayerConfig): Layer;
    static cropping2D(config: Cropping2DLayerConfig): Layer;
    static depthwiseConv2d(config: DepthwiseConv2DLayerConfig): Layer;
    static activation(config: ActivationLayerConfig): Layer;
    static dense(config: DenseLayerConfig): Layer;
    static dropout(config: DropoutLayerConfig): Layer;
    static flatten(config?: LayerConfig): Layer;
    static repeatVector(config: RepeatVectorLayerConfig): Layer;
    static reshape(config: ReshapeLayerConfig): Layer;
    static embedding(config: EmbeddingLayerConfig): Layer;
    static add(config?: LayerConfig): Layer;
    static average(config?: LayerConfig): Layer;
    static concatenate(config?: ConcatenateLayerConfig): Layer;
    static maximum(config?: LayerConfig): Layer;
    static minimum(config?: LayerConfig): Layer;
    static multiply(config?: LayerConfig): Layer;
    static batchNormalization(config: BatchNormalizationLayerConfig): Layer;
    static zeroPadding2d(config?: ZeroPadding2DLayerConfig): Layer;
    static averagePooling1d(config: Pooling1DLayerConfig): Layer;
    static avgPool1d(config: Pooling1DLayerConfig): Layer;
    static avgPooling1d(config: Pooling1DLayerConfig): Layer;
    static averagePooling2d(config: Pooling2DLayerConfig): Layer;
    static avgPool2d(config: Pooling2DLayerConfig): Layer;
    static avgPooling2d(config: Pooling2DLayerConfig): Layer;
    static globalAveragePooling1d(config: LayerConfig): Layer;
    static globalAveragePooling2d(config: GlobalPooling2DLayerConfig): Layer;
    static globalMaxPooling1d(config: LayerConfig): Layer;
    static globalMaxPooling2d(config: GlobalPooling2DLayerConfig): Layer;
    static maxPooling1d(config: Pooling1DLayerConfig): Layer;
    static maxPooling2d(config: Pooling2DLayerConfig): Layer;
    static gru(config: GRULayerConfig): Layer;
    static gruCell(config: GRUCellLayerConfig): RNNCell;
    static lstm(config: LSTMLayerConfig): Layer;
    static lstmCell(config: LSTMCellLayerConfig): RNNCell;
    static simpleRNN(config: SimpleRNNLayerConfig): Layer;
    static simpleRNNCell(config: SimpleRNNCellLayerConfig): RNNCell;
    static rnn(config: RNNLayerConfig): Layer;
    static stackedRNNCells(config: StackedRNNCellsConfig): RNNCell;
    static bidirectional(config: BidirectionalLayerConfig): Wrapper;
    static timeDistributed(config: WrapperLayerConfig): Layer;
}
export declare class ConstraintExports {
    static maxNorm(config: MaxNormConfig): Constraint;
    static unitNorm(config: UnitNormConfig): Constraint;
    static nonNeg(): Constraint;
    static minMaxNorm(config: MinMaxNormConfig): Constraint;
}
export declare class InitializerExports {
    static zeros(): Zeros;
    static ones(): Initializer;
    static constant(config: ConstantConfig): Initializer;
    static randomUniform(config: RandomUniformConfig): Initializer;
    static randomNormal(config: RandomNormalConfig): Initializer;
    static truncatedNormal(config: TruncatedNormalConfig): Initializer;
    static identity(config: IdentityConfig): Initializer;
    static varianceScaling(config: VarianceScalingConfig): Initializer;
    static glorotUniform(config: SeedOnlyInitializerConfig): Initializer;
    static glorotNormal(config: SeedOnlyInitializerConfig): Initializer;
    static heNormal(config: SeedOnlyInitializerConfig): Initializer;
    static leCunNormal(config: SeedOnlyInitializerConfig): Initializer;
    static orthogonal(config: OrthogonalConfig): Initializer;
}
export declare class MetricExports {
    static binaryAccuracy(yTrue: Tensor, yPred: Tensor): Tensor;
    static binaryCrossentropy(yTrue: Tensor, yPred: Tensor): Tensor;
    static categoricalAccuracy(yTrue: Tensor, yPred: Tensor): Tensor;
    static categoricalCrossentropy(yTrue: Tensor, yPred: Tensor): Tensor;
    static cosineProximity(yTrue: Tensor, yPred: Tensor): Tensor;
    meanAbsoluteError(yTrue: Tensor, yPred: Tensor): Tensor;
    meanAbsolutePercentageError(yTrue: Tensor, yPred: Tensor): Tensor;
    MAPE(yTrue: Tensor, yPred: Tensor): Tensor;
    mape(yTrue: Tensor, yPred: Tensor): Tensor;
    static meanSquaredError(yTrue: Tensor, yPred: Tensor): Tensor;
    static MSE(yTrue: Tensor, yPred: Tensor): Tensor;
    static mse(yTrue: Tensor, yPred: Tensor): Tensor;
}
export declare class RegularizerExports {
    static l1l2(config?: L1L2Config): Regularizer;
    static l1(config?: L1Config): Regularizer;
    static l2(config?: L2Config): Regularizer;
}
