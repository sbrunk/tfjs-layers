import { ValueError } from '../errors';
import { pyListRepeat } from './generic_utils';
import { isInteger, max } from './math_utils';
export function normalizeArray(value, n, name) {
    if (typeof value === 'number') {
        return pyListRepeat(value, n);
    }
    else {
        if (value.length !== n) {
            throw new ValueError("The " + name + " argument must be a tuple of " + n + " integers. Received: " +
                (value.length + " elements."));
        }
        for (var i = 0; i < n; ++i) {
            var singleValue = value[i];
            if (!isInteger(singleValue)) {
                throw new ValueError("The " + name + " argument must be a tuple of " + n + " integers. Received: " +
                    (JSON.stringify(value) + " including a non-integer number ") +
                    ("" + singleValue));
            }
        }
        return value;
    }
}
export function convOutputLength(inputLength, fliterSize, padding, stride, dilation) {
    if (dilation === void 0) { dilation = 1; }
    if (inputLength == null) {
        return inputLength;
    }
    var dilatedFilterSize = fliterSize + (fliterSize - 1) * (dilation - 1);
    var outputLength;
    if (padding === 'same') {
        outputLength = inputLength;
    }
    else {
        outputLength = inputLength - dilatedFilterSize + 1;
    }
    return Math.floor((outputLength + stride - 1) / stride);
}
export function deconvLength(dimSize, strideSize, kernelSize, padding) {
    if (dimSize == null) {
        return null;
    }
    if (padding === 'valid') {
        dimSize = dimSize * strideSize + max([kernelSize - strideSize, 0]);
    }
    else if (padding === 'same') {
        dimSize = dimSize * strideSize;
    }
    else {
        throw new ValueError("Unsupport padding mode: " + padding + ".");
    }
    return dimSize;
}
//# sourceMappingURL=conv_utils.js.map