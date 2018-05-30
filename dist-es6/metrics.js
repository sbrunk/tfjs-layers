import * as tfc from '@tensorflow/tfjs-core';
import { tidy } from '@tensorflow/tfjs-core';
import * as K from './backend/tfjs_backend';
import { NotImplementedError, ValueError } from './errors';
import { categoricalCrossentropy as categoricalCrossentropyLoss, cosineProximity, meanAbsoluteError, meanAbsolutePercentageError, meanSquaredError, sparseCategoricalCrossentropy as sparseCategoricalCrossentropyLoss } from './losses';
export function binaryAccuracy(yTrue, yPred) {
    return tidy(function () {
        var threshold = K.scalarTimesArray(K.getScalar(0.5), tfc.onesLike(yPred));
        var yPredThresholded = K.cast(tfc.greater(yPred, threshold), yTrue.dtype);
        return tfc.mean(tfc.equal(yTrue, yPredThresholded), -1);
    });
}
export function categoricalAccuracy(yTrue, yPred) {
    return tidy(function () { return K.cast(tfc.equal(tfc.argMax(yTrue, -1), tfc.argMax(yPred, -1)), 'float32'); });
}
export function binaryCrossentropy(yTrue, yPred) {
    return tidy(function () { return tfc.mean(K.binaryCrossentropy(yTrue, yPred), -1); });
}
export function sparseCategoricalAccuracy(yTrue, yPred) {
    throw new NotImplementedError();
}
export function topKCategoricalAccuracy(yTrue, yPred) {
    throw new NotImplementedError();
}
export function sparseTopKCategoricalAccuracy(yTrue, yPred) {
    throw new NotImplementedError();
}
export var mse = meanSquaredError;
export var MSE = meanSquaredError;
export var mae = meanAbsoluteError;
export var MAE = meanAbsoluteError;
export var mape = meanAbsolutePercentageError;
export var MAPE = meanAbsolutePercentageError;
export var categoricalCrossentropy = categoricalCrossentropyLoss;
export var cosine = cosineProximity;
export var sparseCategoricalCrossentropy = sparseCategoricalCrossentropyLoss;
export function get(identifier) {
    var metricsMap = {
        binaryAccuracy: binaryAccuracy,
        categoricalAccuracy: categoricalAccuracy,
        categoricalCrossentropy: categoricalCrossentropy,
        sparseCategoricalCrossentropy: sparseCategoricalCrossentropy,
        mse: mse,
        MSE: MSE,
        mae: mae,
        MAE: MAE,
        mape: mape,
        MAPE: MAPE,
        cosine: cosine,
    };
    if (typeof identifier === 'string' && identifier in metricsMap) {
        return metricsMap[identifier];
    }
    else if (typeof identifier !== 'string' && identifier != null) {
        return identifier;
    }
    else {
        throw new ValueError("Unknown metric " + identifier);
    }
}
//# sourceMappingURL=metrics.js.map