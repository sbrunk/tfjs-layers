var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
import * as tfc from '@tensorflow/tfjs-core';
import { serialization, tidy } from '@tensorflow/tfjs-core';
import { imageDataFormat } from '../backend/common';
import * as K from '../backend/tfjs_backend';
import { checkDataFormat } from '../common';
import { getConstraint } from '../constraints';
import { ValueError } from '../errors';
import { getInitializer } from '../initializers';
import { getRegularizer } from '../regularizers';
import { convOutputLength } from '../utils/conv_utils';
import { getExactlyOneShape, getExactlyOneTensor } from '../utils/generic_utils';
import { Conv2D, preprocessConv2DInput } from './convolutional';
export function depthwiseConv2d(x, depthwiseKernel, strides, padding, dataFormat, dilationRate) {
    if (strides === void 0) { strides = [1, 1]; }
    if (padding === void 0) { padding = 'valid'; }
    return tidy(function () {
        if (dataFormat == null) {
            dataFormat = imageDataFormat();
        }
        checkDataFormat(dataFormat);
        var y = preprocessConv2DInput(x, dataFormat);
        if (K.ndim(x) !== 4) {
            throw new ValueError("Input for depthwiseConv2d is required to be 4-D, but is instead " +
                (K.ndim(x) + "-D"));
        }
        if (K.ndim(depthwiseKernel) !== 4) {
            throw new ValueError("depthwiseKernel is required to be 4-D, but is instead " +
                (K.ndim(depthwiseKernel) + "-D"));
        }
        y = tfc.depthwiseConv2d(y, depthwiseKernel, strides, padding === 'same' ? 'same' : 'valid', 'NHWC', dilationRate);
        if (dataFormat === 'channelsFirst') {
            y = tfc.transpose(y, [0, 3, 1, 2]);
        }
        return y;
    });
}
var DepthwiseConv2D = (function (_super) {
    __extends(DepthwiseConv2D, _super);
    function DepthwiseConv2D(config) {
        var _this = _super.call(this, config) || this;
        _this.depthwiseKernel = null;
        _this.depthMultiplier =
            config.depthMultiplier == null ? 1 : config.depthMultiplier;
        _this.depthwiseInitializer = getInitializer(config.depthwiseInitializer || _this.DEFAULT_KERNEL_INITIALIZER);
        _this.depthwiseConstraint = getConstraint(config.depthwiseConstraint);
        _this.depthwiseRegularizer = getRegularizer(config.depthwiseRegularizer);
        return _this;
    }
    DepthwiseConv2D.prototype.build = function (inputShape) {
        inputShape = getExactlyOneShape(inputShape);
        if (inputShape.length < 4) {
            throw new ValueError("Inputs to DepthwiseConv2D should have rank 4. " +
                ("Received input shape: " + JSON.stringify(inputShape) + "."));
        }
        var channelAxis = this.dataFormat === 'channelsFirst' ? 1 : 3;
        if (inputShape[channelAxis] == null || inputShape[channelAxis] < 0) {
            throw new ValueError('The channel dimension of the inputs to DepthwiseConv2D should ' +
                ("be defined, but is not (" + inputShape[channelAxis] + ")."));
        }
        var inputDim = inputShape[channelAxis];
        var depthwiseKernelShape = [
            this.kernelSize[0], this.kernelSize[1], inputDim, this.depthMultiplier
        ];
        this.depthwiseKernel = this.addWeight('depthwise_kernel', depthwiseKernelShape, null, this.depthwiseInitializer, this.depthwiseRegularizer, true, this.depthwiseConstraint);
        if (this.useBias) {
            this.bias = this.addWeight('bias', [inputDim * this.depthMultiplier], null, this.biasInitializer, this.biasRegularizer, true, this.biasConstraint);
        }
        else {
            this.bias = null;
        }
        this.built = true;
    };
    DepthwiseConv2D.prototype.call = function (inputs, kwargs) {
        var _this = this;
        return tidy(function () {
            inputs = getExactlyOneTensor(inputs);
            var outputs = depthwiseConv2d(inputs, _this.depthwiseKernel.read(), _this.strides, _this.padding, _this.dataFormat, null);
            if (_this.useBias) {
                outputs = K.biasAdd(outputs, _this.bias.read(), _this.dataFormat);
            }
            if (_this.activation != null) {
                outputs = _this.activation.apply(outputs);
            }
            return outputs;
        });
    };
    DepthwiseConv2D.prototype.computeOutputShape = function (inputShape) {
        inputShape = getExactlyOneShape(inputShape);
        var rows = this.dataFormat === 'channelsFirst' ? inputShape[2] : inputShape[1];
        var cols = this.dataFormat === 'channelsFirst' ? inputShape[3] : inputShape[2];
        var outFilters = this.dataFormat === 'channelsFirst' ?
            inputShape[1] * this.depthMultiplier :
            inputShape[3] * this.depthMultiplier;
        var outRows = convOutputLength(rows, this.kernelSize[0], this.padding, this.strides[0]);
        var outCols = convOutputLength(cols, this.kernelSize[1], this.padding, this.strides[1]);
        if (this.dataFormat === 'channelsFirst') {
            return [inputShape[0], outFilters, outRows, outCols];
        }
        else {
            return [inputShape[0], outRows, outCols, outFilters];
        }
    };
    DepthwiseConv2D.className = 'DepthwiseConv2D';
    return DepthwiseConv2D;
}(Conv2D));
export { DepthwiseConv2D };
serialization.SerializationMap.register(DepthwiseConv2D);
//# sourceMappingURL=convolutional_depthwise.js.map