import * as tfc from '@tensorflow/tfjs-core';
import { tidy } from '@tensorflow/tfjs-core';
import * as K from './backend/tfjs_backend';
import { ValueError } from './errors';
export function meanSquaredError(yTrue, yPred) {
    return tidy(function () { return tfc.mean(K.square(tfc.sub(yPred, yTrue)), -1); });
}
export function meanAbsoluteError(yTrue, yPred) {
    return tidy(function () { return tfc.mean(tfc.abs(tfc.sub(yPred, yTrue)), -1); });
}
export function meanAbsolutePercentageError(yTrue, yPred) {
    return tidy(function () {
        var diff = tfc.sub(yTrue, yPred);
        var clippedTrue = tfc.clipByValue(tfc.abs(yTrue), K.epsilon(), Number.MAX_VALUE);
        var absResult = tfc.abs(tfc.div(diff, clippedTrue));
        return K.scalarTimesArray(K.getScalar(100.0), tfc.mean(absResult, -1));
    });
}
export function meanSquaredLogarithmicError(yTrue, yPred) {
    return tidy(function () {
        var one = K.getScalar(1.0);
        var clippedPred = tfc.clipByValue(yPred, K.epsilon(), Number.MAX_VALUE);
        var firstLog = tfc.log(K.scalarPlusArray(one, clippedPred));
        var clippedTrue = tfc.clipByValue(yTrue, K.epsilon(), Number.MAX_VALUE);
        var secondLog = tfc.log(K.scalarPlusArray(one, clippedTrue));
        return tfc.mean(K.square(tfc.sub(firstLog, secondLog)), -1);
    });
}
export function squaredHinge(yTrue, yPred) {
    return tidy(function () {
        var zeroTensor = K.getScalar(0.0);
        var one = K.getScalar(1.0);
        var maxResult = tfc.maximum(zeroTensor, tfc.sub(one, tfc.mul(yTrue, yPred)));
        return tfc.mean(K.square(maxResult), -1);
    });
}
export function hinge(yTrue, yPred) {
    return tidy(function () {
        var zeroTensor = K.getScalar(0.0);
        var one = K.getScalar(1.0);
        var maxResult = tfc.maximum(zeroTensor, tfc.sub(one, tfc.mul(yTrue, yPred)));
        return tfc.mean(maxResult, -1);
    });
}
export function categoricalHinge(yTrue, yPred) {
    return tidy(function () {
        var zeroTensor = K.getScalar(0.0);
        var one = K.getScalar(1.0);
        var pos = tfc.sum(tfc.mul(yTrue, yPred), -1);
        var neg = tfc.max(tfc.mul(tfc.sub(one, yTrue), yPred), -1);
        return tfc.maximum(zeroTensor, K.scalarPlusArray(one, tfc.sub(neg, pos)));
    });
}
export function logcosh(yTrue, yPred) {
    return tidy(function () {
        var log2 = K.getScalar(Math.log(2.0));
        var predictionDiff = tfc.sub(yPred, yTrue);
        var logcoshResult = tfc.sub(tfc.add(predictionDiff, tfc.softplus(K.scalarTimesArray(K.getScalar(-2.0), predictionDiff))), log2);
        return tfc.mean(logcoshResult, -1);
    });
}
export function categoricalCrossentropy(yTrue, yPred) {
    return tidy(function () { return K.categoricalCrossentropy(yTrue, yPred); });
}
export function sparseCategoricalCrossentropy(yTrue, yPred) {
    return tidy(function () { return K.sparseCategoricalCrossentropy(yTrue, yPred); });
}
export function binaryCrossentropy(yTrue, yPred) {
    return tidy(function () { return tfc.mean(K.binaryCrossentropy(yTrue, yPred), -1); });
}
export function kullbackLeiblerDivergence(yTrue, yPred) {
    return tidy(function () {
        var clippedTrue = tfc.clipByValue(yTrue, K.epsilon(), 1);
        var clippedPred = tfc.clipByValue(yPred, K.epsilon(), 1);
        return tfc.sum(tfc.mul(yTrue, tfc.log(tfc.div(clippedTrue, clippedPred))), -1);
    });
}
export function poisson(yTrue, yPred) {
    return tidy(function () {
        var logPred = tfc.log(K.scalarPlusArray(K.getScalar(K.epsilon()), yPred));
        return tfc.mean(tfc.sub(yPred, tfc.mul(yTrue, logPred)), -1);
    });
}
export function cosineProximity(yTrue, yPred) {
    return tidy(function () {
        var trueNormalized = K.l2Normalize(yTrue, -1);
        var predNormalized = K.l2Normalize(yPred, -1);
        var trueXPred = tfc.mul(trueNormalized, predNormalized);
        return tfc.neg(tfc.sum(trueXPred, -1));
    });
}
export var mse = meanSquaredError;
export var MSE = meanSquaredError;
export var mae = meanAbsoluteError;
export var MAE = meanAbsoluteError;
export var mape = meanAbsolutePercentageError;
export var MAPE = meanAbsolutePercentageError;
export var msle = meanSquaredLogarithmicError;
export var MSLE = meanSquaredLogarithmicError;
export var kld = kullbackLeiblerDivergence;
export var KLD = kullbackLeiblerDivergence;
export var cosine = cosineProximity;
export function get(identifierOrFn) {
    var lossesMap = {
        meanSquaredError: meanSquaredError,
        meanAbsoluteError: meanAbsoluteError,
        meanAbsolutePercentageError: meanAbsolutePercentageError,
        meanSquaredLogarithmicError: meanSquaredLogarithmicError,
        squaredHinge: squaredHinge,
        hinge: hinge,
        categoricalHinge: categoricalHinge,
        logcosh: logcosh,
        categoricalCrossentropy: categoricalCrossentropy,
        sparseCategoricalCrossentropy: sparseCategoricalCrossentropy,
        binaryCrossentropy: binaryCrossentropy,
        kullbackLeiblerDivergence: kullbackLeiblerDivergence,
        poisson: poisson,
        cosineProximity: cosineProximity
    };
    if (typeof identifierOrFn === 'string') {
        if (identifierOrFn in lossesMap) {
            return lossesMap[identifierOrFn];
        }
        throw new ValueError("Unknown loss " + identifierOrFn);
    }
    else {
        return identifierOrFn;
    }
}
//# sourceMappingURL=losses.js.map