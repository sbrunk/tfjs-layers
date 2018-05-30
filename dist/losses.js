"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tfc = require("@tensorflow/tfjs-core");
var tfjs_core_1 = require("@tensorflow/tfjs-core");
var K = require("./backend/tfjs_backend");
var errors_1 = require("./errors");
function meanSquaredError(yTrue, yPred) {
    return tfjs_core_1.tidy(function () { return tfc.mean(K.square(tfc.sub(yPred, yTrue)), -1); });
}
exports.meanSquaredError = meanSquaredError;
function meanAbsoluteError(yTrue, yPred) {
    return tfjs_core_1.tidy(function () { return tfc.mean(tfc.abs(tfc.sub(yPred, yTrue)), -1); });
}
exports.meanAbsoluteError = meanAbsoluteError;
function meanAbsolutePercentageError(yTrue, yPred) {
    return tfjs_core_1.tidy(function () {
        var diff = tfc.sub(yTrue, yPred);
        var clippedTrue = tfc.clipByValue(tfc.abs(yTrue), K.epsilon(), Number.MAX_VALUE);
        var absResult = tfc.abs(tfc.div(diff, clippedTrue));
        return K.scalarTimesArray(K.getScalar(100.0), tfc.mean(absResult, -1));
    });
}
exports.meanAbsolutePercentageError = meanAbsolutePercentageError;
function meanSquaredLogarithmicError(yTrue, yPred) {
    return tfjs_core_1.tidy(function () {
        var one = K.getScalar(1.0);
        var clippedPred = tfc.clipByValue(yPred, K.epsilon(), Number.MAX_VALUE);
        var firstLog = tfc.log(K.scalarPlusArray(one, clippedPred));
        var clippedTrue = tfc.clipByValue(yTrue, K.epsilon(), Number.MAX_VALUE);
        var secondLog = tfc.log(K.scalarPlusArray(one, clippedTrue));
        return tfc.mean(K.square(tfc.sub(firstLog, secondLog)), -1);
    });
}
exports.meanSquaredLogarithmicError = meanSquaredLogarithmicError;
function squaredHinge(yTrue, yPred) {
    return tfjs_core_1.tidy(function () {
        var zeroTensor = K.getScalar(0.0);
        var one = K.getScalar(1.0);
        var maxResult = tfc.maximum(zeroTensor, tfc.sub(one, tfc.mul(yTrue, yPred)));
        return tfc.mean(K.square(maxResult), -1);
    });
}
exports.squaredHinge = squaredHinge;
function hinge(yTrue, yPred) {
    return tfjs_core_1.tidy(function () {
        var zeroTensor = K.getScalar(0.0);
        var one = K.getScalar(1.0);
        var maxResult = tfc.maximum(zeroTensor, tfc.sub(one, tfc.mul(yTrue, yPred)));
        return tfc.mean(maxResult, -1);
    });
}
exports.hinge = hinge;
function categoricalHinge(yTrue, yPred) {
    return tfjs_core_1.tidy(function () {
        var zeroTensor = K.getScalar(0.0);
        var one = K.getScalar(1.0);
        var pos = tfc.sum(tfc.mul(yTrue, yPred), -1);
        var neg = tfc.max(tfc.mul(tfc.sub(one, yTrue), yPred), -1);
        return tfc.maximum(zeroTensor, K.scalarPlusArray(one, tfc.sub(neg, pos)));
    });
}
exports.categoricalHinge = categoricalHinge;
function logcosh(yTrue, yPred) {
    return tfjs_core_1.tidy(function () {
        var log2 = K.getScalar(Math.log(2.0));
        var predictionDiff = tfc.sub(yPred, yTrue);
        var logcoshResult = tfc.sub(tfc.add(predictionDiff, tfc.softplus(K.scalarTimesArray(K.getScalar(-2.0), predictionDiff))), log2);
        return tfc.mean(logcoshResult, -1);
    });
}
exports.logcosh = logcosh;
function categoricalCrossentropy(yTrue, yPred) {
    return tfjs_core_1.tidy(function () { return K.categoricalCrossentropy(yTrue, yPred); });
}
exports.categoricalCrossentropy = categoricalCrossentropy;
function sparseCategoricalCrossentropy(yTrue, yPred) {
    return tfjs_core_1.tidy(function () { return K.sparseCategoricalCrossentropy(yTrue, yPred); });
}
exports.sparseCategoricalCrossentropy = sparseCategoricalCrossentropy;
function binaryCrossentropy(yTrue, yPred) {
    return tfjs_core_1.tidy(function () { return tfc.mean(K.binaryCrossentropy(yTrue, yPred), -1); });
}
exports.binaryCrossentropy = binaryCrossentropy;
function kullbackLeiblerDivergence(yTrue, yPred) {
    return tfjs_core_1.tidy(function () {
        var clippedTrue = tfc.clipByValue(yTrue, K.epsilon(), 1);
        var clippedPred = tfc.clipByValue(yPred, K.epsilon(), 1);
        return tfc.sum(tfc.mul(yTrue, tfc.log(tfc.div(clippedTrue, clippedPred))), -1);
    });
}
exports.kullbackLeiblerDivergence = kullbackLeiblerDivergence;
function poisson(yTrue, yPred) {
    return tfjs_core_1.tidy(function () {
        var logPred = tfc.log(K.scalarPlusArray(K.getScalar(K.epsilon()), yPred));
        return tfc.mean(tfc.sub(yPred, tfc.mul(yTrue, logPred)), -1);
    });
}
exports.poisson = poisson;
function cosineProximity(yTrue, yPred) {
    return tfjs_core_1.tidy(function () {
        var trueNormalized = K.l2Normalize(yTrue, -1);
        var predNormalized = K.l2Normalize(yPred, -1);
        var trueXPred = tfc.mul(trueNormalized, predNormalized);
        return tfc.neg(tfc.sum(trueXPred, -1));
    });
}
exports.cosineProximity = cosineProximity;
exports.mse = meanSquaredError;
exports.MSE = meanSquaredError;
exports.mae = meanAbsoluteError;
exports.MAE = meanAbsoluteError;
exports.mape = meanAbsolutePercentageError;
exports.MAPE = meanAbsolutePercentageError;
exports.msle = meanSquaredLogarithmicError;
exports.MSLE = meanSquaredLogarithmicError;
exports.kld = kullbackLeiblerDivergence;
exports.KLD = kullbackLeiblerDivergence;
exports.cosine = cosineProximity;
function get(identifierOrFn) {
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
        throw new errors_1.ValueError("Unknown loss " + identifierOrFn);
    }
    else {
        return identifierOrFn;
    }
}
exports.get = get;
//# sourceMappingURL=losses.js.map