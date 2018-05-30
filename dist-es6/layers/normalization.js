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
import { movingAverage, serialization, tidy, util } from '@tensorflow/tfjs-core';
import * as K from '../backend/tfjs_backend';
import { getConstraint, serializeConstraint } from '../constraints';
import { InputSpec, Layer } from '../engine/topology';
import { ValueError } from '../errors';
import { getInitializer, serializeInitializer } from '../initializers';
import { getRegularizer, serializeRegularizer } from '../regularizers';
import * as generic_utils from '../utils/generic_utils';
import { arrayProd, range } from '../utils/math_utils';
var BatchNormalization = (function (_super) {
    __extends(BatchNormalization, _super);
    function BatchNormalization(config) {
        var _this = _super.call(this, config) || this;
        _this.supportsMasking = true;
        _this.axis = config.axis == null ? -1 : config.axis;
        _this.momentum = config.momentum == null ? 0.99 : config.momentum;
        _this.epsilon = config.epsilon == null ? 1e-3 : config.epsilon;
        _this.center = config.center == null ? true : config.center;
        _this.scale = config.scale == null ? true : config.scale;
        _this.betaInitializer = getInitializer(config.betaInitializer || 'zeros');
        _this.gammaInitializer = getInitializer(config.gammaInitializer || 'ones');
        _this.movingMeanInitializer =
            getInitializer(config.movingMeanInitializer || 'zeros');
        _this.movingVarianceInitializer =
            getInitializer(config.movingVarianceInitializer || 'ones');
        _this.betaConstraint = getConstraint(config.betaConstraint);
        _this.gammaConstraint = getConstraint(config.gammaConstraint);
        _this.betaRegularizer = getRegularizer(config.betaRegularizer);
        _this.gammaRegularizer = getRegularizer(config.gammaRegularizer);
        _this.stepCount = 0;
        return _this;
    }
    BatchNormalization.prototype.build = function (inputShape) {
        inputShape = generic_utils.getExactlyOneShape(inputShape);
        var axis = this.axis >= 0 ? this.axis : (this.axis + inputShape.length);
        var dim = inputShape[axis];
        if (dim == null) {
            throw new ValueError("Axis " + axis + " of input tensor should have a defined dimension but " +
                "the layer received an input with shape " +
                (JSON.stringify(inputShape) + "."));
        }
        this.inputSpec =
            [new InputSpec({ ndim: inputShape.length, axes: (_a = {}, _a[axis] = dim, _a) })];
        var shape = [dim];
        if (this.scale) {
            this.gamma = this.addWeight('gamma', shape, null, this.gammaInitializer, this.gammaRegularizer, true, this.gammaConstraint);
        }
        if (this.center) {
            this.beta = this.addWeight('beta', shape, null, this.betaInitializer, this.betaRegularizer, true, this.betaConstraint);
        }
        this.movingMean = this.addWeight('moving_mean', shape, null, this.movingMeanInitializer, null, false);
        this.movingVariance = this.addWeight('moving_variance', shape, null, this.movingVarianceInitializer, null, false);
        this.built = true;
        var _a;
    };
    BatchNormalization.prototype.call = function (inputs, kwargs) {
        var _this = this;
        return tidy(function () {
            var training = kwargs['training'] == null ? false : kwargs['training'];
            var input = generic_utils.getExactlyOneTensor(inputs);
            var inputShape = K.shape(input);
            var ndim = inputShape.length;
            var reductionAxes = range(0, ndim);
            var axis = _this.axis >= 0 ? _this.axis : (_this.axis + ndim);
            reductionAxes.splice(axis, 1);
            var broadcastShape = generic_utils.pyListRepeat(1, ndim);
            broadcastShape[axis] = inputShape[axis];
            var sortedReductionAxes = reductionAxes.slice();
            sortedReductionAxes.sort();
            var needsBroadcasting = !util.arraysEqual(sortedReductionAxes, range(0, ndim).slice(0, ndim - 1));
            var normalizeInference = function () {
                if (needsBroadcasting) {
                    var broadcastMovingMean = _this.movingMean.read().reshape(broadcastShape);
                    var broadcastMovingVariance = _this.movingVariance.read().reshape(broadcastShape);
                    var broadcastBeta = _this.center ? _this.beta.read().reshape(broadcastShape) : null;
                    var broadcastGamma = _this.scale ? _this.gamma.read().reshape(broadcastShape) : null;
                    return K.batchNormalization(input, broadcastMovingMean, broadcastMovingVariance, broadcastBeta, broadcastGamma, _this.epsilon);
                }
                else {
                    return K.batchNormalization(input, _this.movingMean.read(), _this.movingVariance.read(), _this.beta == null ? null : _this.beta.read(), _this.gamma == null ? null : _this.gamma.read(), _this.epsilon);
                }
            };
            if (!training) {
                return normalizeInference();
            }
            var _a = K.normalizeBatchInTraining(input, _this.gamma.read(), _this.beta.read(), reductionAxes, _this.epsilon), normedTraining = _a[0], mean = _a[1], variance = _a[2];
            var sampleSize = arrayProd(reductionAxes.map(function (axis) { return input.shape[axis]; }));
            var varianceDebiased = variance.mul(K.getScalar(sampleSize / (sampleSize - (1 + _this.epsilon))));
            var updateMovingMeanAndVariance = function () {
                _this.stepCount++;
                var newMovingMean = movingAverage(_this.movingMean.read(), mean, _this.momentum, _this.stepCount);
                _this.movingMean.write(newMovingMean);
                var newMovingVariance = movingAverage(_this.movingVariance.read(), varianceDebiased, _this.momentum, _this.stepCount);
                _this.movingVariance.write(newMovingVariance);
            };
            updateMovingMeanAndVariance();
            return normedTraining;
        });
    };
    BatchNormalization.prototype.getConfig = function () {
        var config = {
            axis: this.axis,
            momentum: this.momentum,
            epsilon: this.epsilon,
            center: this.center,
            scale: this.scale,
            betaInitializer: serializeInitializer(this.betaInitializer),
            gammaInitializer: serializeInitializer(this.gammaInitializer),
            movingMeanInitializer: serializeInitializer(this.movingMeanInitializer),
            movingVarianceInitializer: serializeInitializer(this.movingVarianceInitializer),
            betaRegularizer: serializeRegularizer(this.betaRegularizer),
            gammaRegularizer: serializeRegularizer(this.gammaRegularizer),
            betaConstraint: serializeConstraint(this.betaConstraint),
            gammaConstraint: serializeConstraint(this.gammaConstraint)
        };
        var baseConfig = _super.prototype.getConfig.call(this);
        Object.assign(config, baseConfig);
        return config;
    };
    BatchNormalization.className = 'BatchNormalization';
    return BatchNormalization;
}(Layer));
export { BatchNormalization };
serialization.SerializationMap.register(BatchNormalization);
//# sourceMappingURL=normalization.js.map