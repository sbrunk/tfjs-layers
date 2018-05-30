var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
import { doc } from '@tensorflow/tfjs-core';
import { MaxNorm, MinMaxNorm, NonNeg, UnitNorm } from './constraints';
import { Input, InputLayer, Layer } from './engine/topology';
import { Model } from './engine/training';
import { Constant, GlorotNormal, GlorotUniform, HeNormal, Identity, LeCunNormal, Ones, Orthogonal, RandomNormal, RandomUniform, TruncatedNormal, VarianceScaling, Zeros } from './initializers';
import { ELU, LeakyReLU, Softmax, ThresholdedReLU } from './layers/advanced_activations';
import { Conv1D, Conv2D, Conv2DTranspose, Cropping2D, SeparableConv2D } from './layers/convolutional';
import { DepthwiseConv2D } from './layers/convolutional_depthwise';
import { Activation, Dense, Dropout, Flatten, RepeatVector, Reshape } from './layers/core';
import { Embedding } from './layers/embeddings';
import { Add, Average, Concatenate, Maximum, Minimum, Multiply } from './layers/merge';
import { BatchNormalization } from './layers/normalization';
import { ZeroPadding2D } from './layers/padding';
import { AveragePooling1D, AveragePooling2D, GlobalAveragePooling1D, GlobalAveragePooling2D, GlobalMaxPooling1D, GlobalMaxPooling2D, MaxPooling1D, MaxPooling2D } from './layers/pooling';
import { GRU, GRUCell, LSTM, LSTMCell, RNN, RNNCell, SimpleRNN, SimpleRNNCell, StackedRNNCells } from './layers/recurrent';
import { Bidirectional, TimeDistributed } from './layers/wrappers';
import { categoricalCrossentropy, cosineProximity, meanAbsoluteError, meanAbsolutePercentageError, meanSquaredError } from './losses';
import { binaryAccuracy, binaryCrossentropy, categoricalAccuracy } from './metrics';
import { loadModelInternal, Sequential } from './models';
import { l1, L1L2, l2 } from './regularizers';
var ModelExports = (function () {
    function ModelExports() {
    }
    ModelExports.model = function (config) {
        return new Model(config);
    };
    ModelExports.sequential = function (config) {
        return new Sequential(config);
    };
    ModelExports.loadModel = function (pathOrIOHandler) {
        return loadModelInternal(pathOrIOHandler);
    };
    ModelExports.input = function (config) {
        return Input(config);
    };
    __decorate([
        doc({ heading: 'Models', subheading: 'Creation', configParamIndices: [0] })
    ], ModelExports, "model", null);
    __decorate([
        doc({ heading: 'Models', subheading: 'Creation', configParamIndices: [0] })
    ], ModelExports, "sequential", null);
    __decorate([
        doc({
            heading: 'Models',
            subheading: 'Loading',
            useDocsFrom: 'loadModelInternal'
        })
    ], ModelExports, "loadModel", null);
    __decorate([
        doc({
            heading: 'Models',
            subheading: 'Inputs',
            useDocsFrom: 'Input',
            configParamIndices: [0]
        })
    ], ModelExports, "input", null);
    return ModelExports;
}());
export { ModelExports };
var LayerExports = (function () {
    function LayerExports() {
    }
    LayerExports.inputLayer = function (config) {
        return new InputLayer(config);
    };
    LayerExports.elu = function (config) {
        return new ELU(config);
    };
    LayerExports.leakyReLU = function (config) {
        return new LeakyReLU(config);
    };
    LayerExports.softmax = function (config) {
        return new Softmax(config);
    };
    LayerExports.thresholdedReLU = function (config) {
        return new ThresholdedReLU(config);
    };
    LayerExports.conv1d = function (config) {
        return new Conv1D(config);
    };
    LayerExports.conv2d = function (config) {
        return new Conv2D(config);
    };
    LayerExports.conv2dTranspose = function (config) {
        return new Conv2DTranspose(config);
    };
    LayerExports.separableConv2d = function (config) {
        return new SeparableConv2D(config);
    };
    LayerExports.cropping2D = function (config) {
        return new Cropping2D(config);
    };
    LayerExports.depthwiseConv2d = function (config) {
        return new DepthwiseConv2D(config);
    };
    LayerExports.activation = function (config) {
        return new Activation(config);
    };
    LayerExports.dense = function (config) {
        return new Dense(config);
    };
    LayerExports.dropout = function (config) {
        return new Dropout(config);
    };
    LayerExports.flatten = function (config) {
        return new Flatten(config);
    };
    LayerExports.repeatVector = function (config) {
        return new RepeatVector(config);
    };
    LayerExports.reshape = function (config) {
        return new Reshape(config);
    };
    LayerExports.embedding = function (config) {
        return new Embedding(config);
    };
    LayerExports.add = function (config) {
        return new Add(config);
    };
    LayerExports.average = function (config) {
        return new Average(config);
    };
    LayerExports.concatenate = function (config) {
        return new Concatenate(config);
    };
    LayerExports.maximum = function (config) {
        return new Maximum(config);
    };
    LayerExports.minimum = function (config) {
        return new Minimum(config);
    };
    LayerExports.multiply = function (config) {
        return new Multiply(config);
    };
    LayerExports.batchNormalization = function (config) {
        return new BatchNormalization(config);
    };
    LayerExports.zeroPadding2d = function (config) {
        return new ZeroPadding2D(config);
    };
    LayerExports.averagePooling1d = function (config) {
        return new AveragePooling1D(config);
    };
    LayerExports.avgPool1d = function (config) {
        return LayerExports.averagePooling1d(config);
    };
    LayerExports.avgPooling1d = function (config) {
        return LayerExports.averagePooling1d(config);
    };
    LayerExports.averagePooling2d = function (config) {
        return new AveragePooling2D(config);
    };
    LayerExports.avgPool2d = function (config) {
        return LayerExports.averagePooling2d(config);
    };
    LayerExports.avgPooling2d = function (config) {
        return LayerExports.averagePooling2d(config);
    };
    LayerExports.globalAveragePooling1d = function (config) {
        return new GlobalAveragePooling1D(config);
    };
    LayerExports.globalAveragePooling2d = function (config) {
        return new GlobalAveragePooling2D(config);
    };
    LayerExports.globalMaxPooling1d = function (config) {
        return new GlobalMaxPooling1D(config);
    };
    LayerExports.globalMaxPooling2d = function (config) {
        return new GlobalMaxPooling2D(config);
    };
    LayerExports.maxPooling1d = function (config) {
        return new MaxPooling1D(config);
    };
    LayerExports.maxPooling2d = function (config) {
        return new MaxPooling2D(config);
    };
    LayerExports.gru = function (config) {
        return new GRU(config);
    };
    LayerExports.gruCell = function (config) {
        return new GRUCell(config);
    };
    LayerExports.lstm = function (config) {
        return new LSTM(config);
    };
    LayerExports.lstmCell = function (config) {
        return new LSTMCell(config);
    };
    LayerExports.simpleRNN = function (config) {
        return new SimpleRNN(config);
    };
    LayerExports.simpleRNNCell = function (config) {
        return new SimpleRNNCell(config);
    };
    LayerExports.rnn = function (config) {
        return new RNN(config);
    };
    LayerExports.stackedRNNCells = function (config) {
        return new StackedRNNCells(config);
    };
    LayerExports.bidirectional = function (config) {
        return new Bidirectional(config);
    };
    LayerExports.timeDistributed = function (config) {
        return new TimeDistributed(config);
    };
    LayerExports.Layer = Layer;
    LayerExports.RNN = RNN;
    LayerExports.RNNCell = RNNCell;
    LayerExports.input = ModelExports.input;
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Inputs',
            namespace: 'layers',
            useDocsFrom: 'InputLayer',
            configParamIndices: [0]
        })
    ], LayerExports, "inputLayer", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Advanced Activation',
            namespace: 'layers',
            useDocsFrom: 'ELU',
            configParamIndices: [0]
        })
    ], LayerExports, "elu", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Advanced Activation',
            namespace: 'layers',
            useDocsFrom: 'LeakyReLU',
            configParamIndices: [0]
        })
    ], LayerExports, "leakyReLU", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Advanced Activation',
            namespace: 'layers',
            useDocsFrom: 'Softmax',
            configParamIndices: [0]
        })
    ], LayerExports, "softmax", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Advanced Activation',
            namespace: 'layers',
            useDocsFrom: 'ThresholdedReLU',
            configParamIndices: [0]
        })
    ], LayerExports, "thresholdedReLU", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Convolutional',
            namespace: 'layers',
            useDocsFrom: 'Conv1D',
            configParamIndices: [0]
        })
    ], LayerExports, "conv1d", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Convolutional',
            namespace: 'layers',
            useDocsFrom: 'Conv2D',
            configParamIndices: [0]
        })
    ], LayerExports, "conv2d", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Convolutional',
            namespace: 'layers',
            useDocsFrom: 'Conv2DTranspose',
            configParamIndices: [0]
        })
    ], LayerExports, "conv2dTranspose", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Convolutional',
            namespace: 'layers',
            useDocsFrom: 'SeparableConv2D',
            configParamIndices: [0]
        })
    ], LayerExports, "separableConv2d", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Convolutional',
            namespace: 'layers',
            useDocsFrom: 'Cropping2D',
            configParamIndices: [0]
        })
    ], LayerExports, "cropping2D", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Convolutional',
            namespace: 'layers',
            useDocsFrom: 'DepthwiseConv2D',
            configParamIndices: [0]
        })
    ], LayerExports, "depthwiseConv2d", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Basic',
            namespace: 'layers',
            useDocsFrom: 'Activation',
            configParamIndices: [0]
        })
    ], LayerExports, "activation", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Basic',
            namespace: 'layers',
            useDocsFrom: 'Dense',
            configParamIndices: [0]
        })
    ], LayerExports, "dense", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Basic',
            namespace: 'layers',
            useDocsFrom: 'Dropout',
            configParamIndices: [0]
        })
    ], LayerExports, "dropout", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Basic',
            namespace: 'layers',
            useDocsFrom: 'Flatten',
            configParamIndices: [0]
        })
    ], LayerExports, "flatten", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Basic',
            namespace: 'layers',
            useDocsFrom: 'RepeatVector',
            configParamIndices: [0]
        })
    ], LayerExports, "repeatVector", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Basic',
            namespace: 'layers',
            useDocsFrom: 'Reshape',
            configParamIndices: [0]
        })
    ], LayerExports, "reshape", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Basic',
            namespace: 'layers',
            useDocsFrom: 'Embedding',
            configParamIndices: [0]
        })
    ], LayerExports, "embedding", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Merge',
            namespace: 'layers',
            useDocsFrom: 'Add',
            configParamIndices: [0]
        })
    ], LayerExports, "add", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Merge',
            namespace: 'layers',
            useDocsFrom: 'Average',
            configParamIndices: [0]
        })
    ], LayerExports, "average", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Merge',
            namespace: 'layers',
            useDocsFrom: 'Concatenate',
            configParamIndices: [0]
        })
    ], LayerExports, "concatenate", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Merge',
            namespace: 'layers',
            useDocsFrom: 'Maximum',
            configParamIndices: [0]
        })
    ], LayerExports, "maximum", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Merge',
            namespace: 'layers',
            useDocsFrom: 'Minimum',
            configParamIndices: [0]
        })
    ], LayerExports, "minimum", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Merge',
            namespace: 'layers',
            useDocsFrom: 'Multiply',
            configParamIndices: [0]
        })
    ], LayerExports, "multiply", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Normalization',
            namespace: 'layers',
            useDocsFrom: 'BatchNormalization',
            configParamIndices: [0]
        })
    ], LayerExports, "batchNormalization", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Padding',
            namespace: 'layers',
            useDocsFrom: 'ZeroPadding2D',
            configParamIndices: [0]
        })
    ], LayerExports, "zeroPadding2d", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Pooling',
            namespace: 'layers',
            useDocsFrom: 'AveragePooling1D',
            configParamIndices: [0]
        })
    ], LayerExports, "averagePooling1d", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Pooling',
            namespace: 'layers',
            useDocsFrom: 'AveragePooling2D',
            configParamIndices: [0]
        })
    ], LayerExports, "averagePooling2d", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Pooling',
            namespace: 'layers',
            useDocsFrom: 'GlobalAveragePooling1D',
            configParamIndices: [0]
        })
    ], LayerExports, "globalAveragePooling1d", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Pooling',
            namespace: 'layers',
            useDocsFrom: 'GlobalAveragePooling2D',
            configParamIndices: [0]
        })
    ], LayerExports, "globalAveragePooling2d", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Pooling',
            namespace: 'layers',
            useDocsFrom: 'GlobalMaxPooling1D',
            configParamIndices: [0]
        })
    ], LayerExports, "globalMaxPooling1d", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Pooling',
            namespace: 'layers',
            useDocsFrom: 'GlobalMaxPooling2D',
            configParamIndices: [0]
        })
    ], LayerExports, "globalMaxPooling2d", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Pooling',
            namespace: 'layers',
            useDocsFrom: 'MaxPooling1D',
            configParamIndices: [0]
        })
    ], LayerExports, "maxPooling1d", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Pooling',
            namespace: 'layers',
            useDocsFrom: 'MaxPooling2D',
            configParamIndices: [0]
        })
    ], LayerExports, "maxPooling2d", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Recurrent',
            namespace: 'layers',
            useDocsFrom: 'GRU',
            configParamIndices: [0]
        })
    ], LayerExports, "gru", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Recurrent',
            namespace: 'layers',
            useDocsFrom: 'GRUCell',
            configParamIndices: [0]
        })
    ], LayerExports, "gruCell", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Recurrent',
            namespace: 'layers',
            useDocsFrom: 'LSTM',
            configParamIndices: [0]
        })
    ], LayerExports, "lstm", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Recurrent',
            namespace: 'layers',
            useDocsFrom: 'LSTMCell',
            configParamIndices: [0]
        })
    ], LayerExports, "lstmCell", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Recurrent',
            namespace: 'layers',
            useDocsFrom: 'SimpleRNN',
            configParamIndices: [0]
        })
    ], LayerExports, "simpleRNN", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Recurrent',
            namespace: 'layers',
            useDocsFrom: 'SimpleRNNCell',
            configParamIndices: [0]
        })
    ], LayerExports, "simpleRNNCell", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Recurrent',
            namespace: 'layers',
            useDocsFrom: 'RNN',
            configParamIndices: [0]
        })
    ], LayerExports, "rnn", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Recurrent',
            namespace: 'layers',
            useDocsFrom: 'RNN',
            configParamIndices: [0]
        })
    ], LayerExports, "stackedRNNCells", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Wrapper',
            namespace: 'layers',
            useDocsFrom: 'Bidirectional',
            configParamIndices: [0]
        })
    ], LayerExports, "bidirectional", null);
    __decorate([
        doc({
            heading: 'Layers',
            subheading: 'Wrapper',
            namespace: 'layers',
            useDocsFrom: 'TimeDistributed',
            configParamIndices: [0]
        })
    ], LayerExports, "timeDistributed", null);
    return LayerExports;
}());
export { LayerExports };
var ConstraintExports = (function () {
    function ConstraintExports() {
    }
    ConstraintExports.maxNorm = function (config) {
        return new MaxNorm(config);
    };
    ConstraintExports.unitNorm = function (config) {
        return new UnitNorm(config);
    };
    ConstraintExports.nonNeg = function () {
        return new NonNeg();
    };
    ConstraintExports.minMaxNorm = function (config) {
        return new MinMaxNorm(config);
    };
    __decorate([
        doc({
            heading: 'Constraints',
            namespace: 'constraints',
            useDocsFrom: 'MaxNorm',
            configParamIndices: [0]
        })
    ], ConstraintExports, "maxNorm", null);
    __decorate([
        doc({
            heading: 'Constraints',
            namespace: 'constraints',
            useDocsFrom: 'UnitNorm',
            configParamIndices: [0]
        })
    ], ConstraintExports, "unitNorm", null);
    __decorate([
        doc({ heading: 'Constraints', namespace: 'constraints', useDocsFrom: 'NonNeg' })
    ], ConstraintExports, "nonNeg", null);
    __decorate([
        doc({
            heading: 'Constraints',
            namespace: 'constraints',
            useDocsFrom: 'MinMaxNormConfig',
            configParamIndices: [0]
        })
    ], ConstraintExports, "minMaxNorm", null);
    return ConstraintExports;
}());
export { ConstraintExports };
var InitializerExports = (function () {
    function InitializerExports() {
    }
    InitializerExports.zeros = function () {
        return new Zeros();
    };
    InitializerExports.ones = function () {
        return new Ones();
    };
    InitializerExports.constant = function (config) {
        return new Constant(config);
    };
    InitializerExports.randomUniform = function (config) {
        return new RandomUniform(config);
    };
    InitializerExports.randomNormal = function (config) {
        return new RandomNormal(config);
    };
    InitializerExports.truncatedNormal = function (config) {
        return new TruncatedNormal(config);
    };
    InitializerExports.identity = function (config) {
        return new Identity(config);
    };
    InitializerExports.varianceScaling = function (config) {
        return new VarianceScaling(config);
    };
    InitializerExports.glorotUniform = function (config) {
        return new GlorotUniform(config);
    };
    InitializerExports.glorotNormal = function (config) {
        return new GlorotNormal(config);
    };
    InitializerExports.heNormal = function (config) {
        return new HeNormal(config);
    };
    InitializerExports.leCunNormal = function (config) {
        return new LeCunNormal(config);
    };
    InitializerExports.orthogonal = function (config) {
        return new Orthogonal(config);
    };
    __decorate([
        doc({
            heading: 'Initializers',
            namespace: 'initializers',
            useDocsFrom: 'Zeros'
        })
    ], InitializerExports, "zeros", null);
    __decorate([
        doc({ heading: 'Initializers', namespace: 'initializers', useDocsFrom: 'Ones' })
    ], InitializerExports, "ones", null);
    __decorate([
        doc({
            heading: 'Initializers',
            namespace: 'initializers',
            useDocsFrom: 'Constant',
            configParamIndices: [0]
        })
    ], InitializerExports, "constant", null);
    __decorate([
        doc({
            heading: 'Initializers',
            namespace: 'initializers',
            useDocsFrom: 'RandomUniform',
            configParamIndices: [0]
        })
    ], InitializerExports, "randomUniform", null);
    __decorate([
        doc({
            heading: 'Initializers',
            namespace: 'initializers',
            useDocsFrom: 'RandomNormal',
            configParamIndices: [0]
        })
    ], InitializerExports, "randomNormal", null);
    __decorate([
        doc({
            heading: 'Initializers',
            namespace: 'initializers',
            useDocsFrom: 'TruncatedNormal',
            configParamIndices: [0]
        })
    ], InitializerExports, "truncatedNormal", null);
    __decorate([
        doc({
            heading: 'Initializers',
            namespace: 'initializers',
            useDocsFrom: 'Identity',
            configParamIndices: [0]
        })
    ], InitializerExports, "identity", null);
    __decorate([
        doc({
            heading: 'Initializers',
            namespace: 'initializers',
            useDocsFrom: 'VarianceScaling',
            configParamIndices: [0]
        })
    ], InitializerExports, "varianceScaling", null);
    __decorate([
        doc({
            heading: 'Initializers',
            namespace: 'initializers',
            useDocsFrom: 'GlorotUniform',
            configParamIndices: [0]
        })
    ], InitializerExports, "glorotUniform", null);
    __decorate([
        doc({
            heading: 'Initializers',
            namespace: 'initializers',
            useDocsFrom: 'GlorotNormal',
            configParamIndices: [0]
        })
    ], InitializerExports, "glorotNormal", null);
    __decorate([
        doc({
            heading: 'Initializers',
            namespace: 'initializers',
            useDocsFrom: 'HeNormal',
            configParamIndices: [0]
        })
    ], InitializerExports, "heNormal", null);
    __decorate([
        doc({
            heading: 'Initializers',
            namespace: 'initializers',
            useDocsFrom: 'LeCunNormal',
            configParamIndices: [0]
        })
    ], InitializerExports, "leCunNormal", null);
    __decorate([
        doc({
            heading: 'Initializers',
            namespace: 'initializers',
            useDocsFrom: 'Orthogonal',
            configParamIndices: [0]
        })
    ], InitializerExports, "orthogonal", null);
    return InitializerExports;
}());
export { InitializerExports };
var MetricExports = (function () {
    function MetricExports() {
    }
    MetricExports.binaryAccuracy = function (yTrue, yPred) {
        return binaryAccuracy(yTrue, yPred);
    };
    MetricExports.binaryCrossentropy = function (yTrue, yPred) {
        return binaryCrossentropy(yTrue, yPred);
    };
    MetricExports.categoricalAccuracy = function (yTrue, yPred) {
        return categoricalAccuracy(yTrue, yPred);
    };
    MetricExports.categoricalCrossentropy = function (yTrue, yPred) {
        return categoricalCrossentropy(yTrue, yPred);
    };
    MetricExports.cosineProximity = function (yTrue, yPred) {
        return cosineProximity(yTrue, yPred);
    };
    MetricExports.prototype.meanAbsoluteError = function (yTrue, yPred) {
        return meanAbsoluteError(yTrue, yPred);
    };
    MetricExports.prototype.meanAbsolutePercentageError = function (yTrue, yPred) {
        return meanAbsolutePercentageError(yTrue, yPred);
    };
    MetricExports.prototype.MAPE = function (yTrue, yPred) {
        return meanAbsolutePercentageError(yTrue, yPred);
    };
    MetricExports.prototype.mape = function (yTrue, yPred) {
        return meanAbsolutePercentageError(yTrue, yPred);
    };
    MetricExports.meanSquaredError = function (yTrue, yPred) {
        return meanSquaredError(yTrue, yPred);
    };
    MetricExports.MSE = function (yTrue, yPred) {
        return meanSquaredError(yTrue, yPred);
    };
    MetricExports.mse = function (yTrue, yPred) {
        return meanSquaredError(yTrue, yPred);
    };
    __decorate([
        doc({
            heading: 'Metrics',
            namespace: 'metrics',
            useDocsFrom: 'meanAbsoluteError'
        })
    ], MetricExports.prototype, "meanAbsoluteError", null);
    __decorate([
        doc({
            heading: 'Metrics',
            namespace: 'metrics',
            useDocsFrom: 'meanAbsolutePercentageError'
        })
    ], MetricExports.prototype, "meanAbsolutePercentageError", null);
    __decorate([
        doc({ heading: 'Metrics', namespace: 'metrics', useDocsFrom: 'binaryAccuracy' })
    ], MetricExports, "binaryAccuracy", null);
    __decorate([
        doc({
            heading: 'Metrics',
            namespace: 'metrics',
            useDocsFrom: 'binaryCrossentropy'
        })
    ], MetricExports, "binaryCrossentropy", null);
    __decorate([
        doc({
            heading: 'Metrics',
            namespace: 'metrics',
            useDocsFrom: 'categoricalAccuracy'
        })
    ], MetricExports, "categoricalAccuracy", null);
    __decorate([
        doc({
            heading: 'Metrics',
            namespace: 'metrics',
            useDocsFrom: 'categoricalCrossentropy'
        })
    ], MetricExports, "categoricalCrossentropy", null);
    __decorate([
        doc({
            heading: 'Metrics',
            namespace: 'metrics',
            useDocsFrom: 'cosineProximity'
        })
    ], MetricExports, "cosineProximity", null);
    __decorate([
        doc({
            heading: 'Metrics',
            namespace: 'metrics',
            useDocsFrom: 'meanSquaredError'
        })
    ], MetricExports, "meanSquaredError", null);
    return MetricExports;
}());
export { MetricExports };
var RegularizerExports = (function () {
    function RegularizerExports() {
    }
    RegularizerExports.l1l2 = function (config) {
        return new L1L2(config);
    };
    RegularizerExports.l1 = function (config) {
        return l1(config);
    };
    RegularizerExports.l2 = function (config) {
        return l2(config);
    };
    __decorate([
        doc({ heading: 'Regularizers', namespace: 'regularizers', useDocsFrom: 'L1L2' })
    ], RegularizerExports, "l1l2", null);
    __decorate([
        doc({ heading: 'Regularizers', namespace: 'regularizers', useDocsFrom: 'L1L2' })
    ], RegularizerExports, "l1", null);
    __decorate([
        doc({ heading: 'Regularizers', namespace: 'regularizers', useDocsFrom: 'L1L2' })
    ], RegularizerExports, "l2", null);
    return RegularizerExports;
}());
export { RegularizerExports };
//# sourceMappingURL=exports.js.map