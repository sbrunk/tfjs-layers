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
import { checkDataFormat, checkPaddingMode, checkPoolMode } from '../common';
import { InputSpec } from '../engine/topology';
import { Layer } from '../engine/topology';
import { NotImplementedError } from '../errors';
import { convOutputLength } from '../utils/conv_utils';
import * as generic_utils from '../utils/generic_utils';
import { preprocessConv2DInput } from './convolutional';
export function pool2d(x, poolSize, strides, padding, dataFormat, poolMode) {
    return tidy(function () {
        checkDataFormat(dataFormat);
        checkPoolMode(poolMode);
        checkPaddingMode(padding);
        if (strides == null) {
            strides = [1, 1];
        }
        if (padding == null) {
            padding = 'valid';
        }
        if (dataFormat == null) {
            dataFormat = imageDataFormat();
        }
        if (poolMode == null) {
            poolMode = 'max';
        }
        x = preprocessConv2DInput(x, dataFormat);
        var y;
        var paddingString = (padding === 'same') ? 'same' : 'valid';
        if (poolMode === 'max') {
            y = tfc.maxPool(x, poolSize, strides, paddingString);
        }
        else {
            y = tfc.avgPool(x, poolSize, strides, paddingString);
        }
        if (dataFormat === 'channelsFirst') {
            y = tfc.transpose(y, [0, 3, 1, 2]);
        }
        return y;
    });
}
var Pooling1D = (function (_super) {
    __extends(Pooling1D, _super);
    function Pooling1D(config) {
        var _this = this;
        if (config.poolSize == null) {
            config.poolSize = 2;
        }
        _this = _super.call(this, config) || this;
        _this.poolSize = [config.poolSize];
        _this.strides = config.strides == null ? _this.poolSize : [config.strides];
        _this.padding = config.padding == null ? 'valid' : config.padding;
        checkPaddingMode(_this.padding);
        _this.inputSpec = [new InputSpec({ ndim: 3 })];
        return _this;
    }
    Pooling1D.prototype.computeOutputShape = function (inputShape) {
        inputShape = generic_utils.getExactlyOneShape(inputShape);
        var length = convOutputLength(inputShape[1], this.poolSize[0], this.padding, this.strides[0]);
        return [inputShape[0], length, inputShape[2]];
    };
    Pooling1D.prototype.call = function (inputs, kwargs) {
        var _this = this;
        return tidy(function () {
            _this.invokeCallHook(inputs, kwargs);
            inputs = K.expandDims(generic_utils.getExactlyOneTensor(inputs), 2);
            var output = _this.poolingFunction(generic_utils.getExactlyOneTensor(inputs), [_this.poolSize[0], 1], [_this.strides[0], 1], _this.padding, 'channelsLast');
            return tfc.squeeze(output, [2]);
        });
    };
    Pooling1D.prototype.getConfig = function () {
        var config = {
            poolSize: this.poolSize,
            padding: this.padding,
            strides: this.strides,
        };
        var baseConfig = _super.prototype.getConfig.call(this);
        Object.assign(config, baseConfig);
        return config;
    };
    return Pooling1D;
}(Layer));
export { Pooling1D };
var MaxPooling1D = (function (_super) {
    __extends(MaxPooling1D, _super);
    function MaxPooling1D(config) {
        return _super.call(this, config) || this;
    }
    MaxPooling1D.prototype.poolingFunction = function (inputs, poolSize, strides, padding, dataFormat) {
        checkDataFormat(dataFormat);
        checkPaddingMode(padding);
        return pool2d(inputs, poolSize, strides, padding, dataFormat, 'max');
    };
    MaxPooling1D.className = 'MaxPooling1D';
    return MaxPooling1D;
}(Pooling1D));
export { MaxPooling1D };
serialization.SerializationMap.register(MaxPooling1D);
var AveragePooling1D = (function (_super) {
    __extends(AveragePooling1D, _super);
    function AveragePooling1D(config) {
        return _super.call(this, config) || this;
    }
    AveragePooling1D.prototype.poolingFunction = function (inputs, poolSize, strides, padding, dataFormat) {
        checkDataFormat(dataFormat);
        checkPaddingMode(padding);
        return pool2d(inputs, poolSize, strides, padding, dataFormat, 'avg');
    };
    AveragePooling1D.className = 'AveragePooling1D';
    return AveragePooling1D;
}(Pooling1D));
export { AveragePooling1D };
serialization.SerializationMap.register(AveragePooling1D);
var Pooling2D = (function (_super) {
    __extends(Pooling2D, _super);
    function Pooling2D(config) {
        var _this = this;
        if (config.poolSize == null) {
            config.poolSize = [2, 2];
        }
        _this = _super.call(this, config) || this;
        _this.poolSize = Array.isArray(config.poolSize) ?
            config.poolSize :
            [config.poolSize, config.poolSize];
        _this.strides = config.strides == null ? _this.poolSize : config.strides;
        _this.padding = config.padding == null ? 'valid' : config.padding;
        _this.dataFormat =
            config.dataFormat == null ? 'channelsLast' : config.dataFormat;
        checkDataFormat(_this.dataFormat);
        checkPaddingMode(_this.padding);
        _this.inputSpec = [new InputSpec({ ndim: 4 })];
        return _this;
    }
    Pooling2D.prototype.computeOutputShape = function (inputShape) {
        inputShape = generic_utils.getExactlyOneShape(inputShape);
        var rows = this.dataFormat === 'channelsFirst' ? inputShape[2] : inputShape[1];
        var cols = this.dataFormat === 'channelsFirst' ? inputShape[3] : inputShape[2];
        rows =
            convOutputLength(rows, this.poolSize[0], this.padding, this.strides[0]);
        cols =
            convOutputLength(cols, this.poolSize[1], this.padding, this.strides[1]);
        if (this.dataFormat === 'channelsFirst') {
            return [inputShape[0], inputShape[1], rows, cols];
        }
        else {
            return [inputShape[0], rows, cols, inputShape[3]];
        }
    };
    Pooling2D.prototype.call = function (inputs, kwargs) {
        var _this = this;
        return tidy(function () {
            _this.invokeCallHook(inputs, kwargs);
            return _this.poolingFunction(generic_utils.getExactlyOneTensor(inputs), _this.poolSize, _this.strides, _this.padding, _this.dataFormat);
        });
    };
    Pooling2D.prototype.getConfig = function () {
        var config = {
            poolSize: this.poolSize,
            padding: this.padding,
            strides: this.strides,
            dataFormat: this.dataFormat
        };
        var baseConfig = _super.prototype.getConfig.call(this);
        Object.assign(config, baseConfig);
        return config;
    };
    return Pooling2D;
}(Layer));
export { Pooling2D };
var MaxPooling2D = (function (_super) {
    __extends(MaxPooling2D, _super);
    function MaxPooling2D(config) {
        return _super.call(this, config) || this;
    }
    MaxPooling2D.prototype.poolingFunction = function (inputs, poolSize, strides, padding, dataFormat) {
        checkDataFormat(dataFormat);
        checkPaddingMode(padding);
        return pool2d(inputs, poolSize, strides, padding, dataFormat, 'max');
    };
    MaxPooling2D.className = 'MaxPooling2D';
    return MaxPooling2D;
}(Pooling2D));
export { MaxPooling2D };
serialization.SerializationMap.register(MaxPooling2D);
var AveragePooling2D = (function (_super) {
    __extends(AveragePooling2D, _super);
    function AveragePooling2D(config) {
        return _super.call(this, config) || this;
    }
    AveragePooling2D.prototype.poolingFunction = function (inputs, poolSize, strides, padding, dataFormat) {
        checkDataFormat(dataFormat);
        checkPaddingMode(padding);
        return pool2d(inputs, poolSize, strides, padding, dataFormat, 'avg');
    };
    AveragePooling2D.className = 'AveragePooling2D';
    return AveragePooling2D;
}(Pooling2D));
export { AveragePooling2D };
serialization.SerializationMap.register(AveragePooling2D);
var GlobalPooling1D = (function (_super) {
    __extends(GlobalPooling1D, _super);
    function GlobalPooling1D(config) {
        var _this = _super.call(this, config) || this;
        _this.inputSpec = [new InputSpec({ ndim: 3 })];
        return _this;
    }
    GlobalPooling1D.prototype.computeOutputShape = function (inputShape) {
        return [inputShape[0], inputShape[2]];
    };
    GlobalPooling1D.prototype.call = function (inputs, kwargs) {
        throw new NotImplementedError();
    };
    return GlobalPooling1D;
}(Layer));
export { GlobalPooling1D };
var GlobalAveragePooling1D = (function (_super) {
    __extends(GlobalAveragePooling1D, _super);
    function GlobalAveragePooling1D(config) {
        return _super.call(this, config) || this;
    }
    GlobalAveragePooling1D.prototype.call = function (inputs, kwargs) {
        return tidy(function () {
            var input = generic_utils.getExactlyOneTensor(inputs);
            return tfc.mean(input, 1);
        });
    };
    GlobalAveragePooling1D.className = 'GlobalAveragePooling1D';
    return GlobalAveragePooling1D;
}(GlobalPooling1D));
export { GlobalAveragePooling1D };
serialization.SerializationMap.register(GlobalAveragePooling1D);
var GlobalMaxPooling1D = (function (_super) {
    __extends(GlobalMaxPooling1D, _super);
    function GlobalMaxPooling1D(config) {
        return _super.call(this, config) || this;
    }
    GlobalMaxPooling1D.prototype.call = function (inputs, kwargs) {
        return tidy(function () {
            var input = generic_utils.getExactlyOneTensor(inputs);
            return tfc.max(input, 1);
        });
    };
    GlobalMaxPooling1D.className = 'GlobalMaxPooling1D';
    return GlobalMaxPooling1D;
}(GlobalPooling1D));
export { GlobalMaxPooling1D };
serialization.SerializationMap.register(GlobalMaxPooling1D);
var GlobalPooling2D = (function (_super) {
    __extends(GlobalPooling2D, _super);
    function GlobalPooling2D(config) {
        var _this = _super.call(this, config) || this;
        _this.dataFormat =
            config.dataFormat == null ? 'channelsLast' : config.dataFormat;
        checkDataFormat(_this.dataFormat);
        _this.inputSpec = [new InputSpec({ ndim: 4 })];
        return _this;
    }
    GlobalPooling2D.prototype.computeOutputShape = function (inputShape) {
        inputShape = inputShape;
        if (this.dataFormat === 'channelsLast') {
            return [inputShape[0], inputShape[3]];
        }
        else {
            return [inputShape[0], inputShape[1]];
        }
    };
    GlobalPooling2D.prototype.call = function (inputs, kwargs) {
        throw new NotImplementedError();
    };
    GlobalPooling2D.prototype.getConfig = function () {
        var config = { dataFormat: this.dataFormat };
        var baseConfig = _super.prototype.getConfig.call(this);
        Object.assign(config, baseConfig);
        return config;
    };
    return GlobalPooling2D;
}(Layer));
export { GlobalPooling2D };
var GlobalAveragePooling2D = (function (_super) {
    __extends(GlobalAveragePooling2D, _super);
    function GlobalAveragePooling2D() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    GlobalAveragePooling2D.prototype.call = function (inputs, kwargs) {
        var _this = this;
        return tidy(function () {
            var input = generic_utils.getExactlyOneTensor(inputs);
            if (_this.dataFormat === 'channelsLast') {
                return tfc.mean(input, [1, 2]);
            }
            else {
                return tfc.mean(input, [2, 3]);
            }
        });
    };
    GlobalAveragePooling2D.className = 'GlobalAveragePooling2D';
    return GlobalAveragePooling2D;
}(GlobalPooling2D));
export { GlobalAveragePooling2D };
serialization.SerializationMap.register(GlobalAveragePooling2D);
var GlobalMaxPooling2D = (function (_super) {
    __extends(GlobalMaxPooling2D, _super);
    function GlobalMaxPooling2D() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    GlobalMaxPooling2D.prototype.call = function (inputs, kwargs) {
        var _this = this;
        return tidy(function () {
            var input = generic_utils.getExactlyOneTensor(inputs);
            if (_this.dataFormat === 'channelsLast') {
                return tfc.max(input, [1, 2]);
            }
            else {
                return tfc.max(input, [2, 3]);
            }
        });
    };
    GlobalMaxPooling2D.className = 'GlobalMaxPooling2D';
    return GlobalMaxPooling2D;
}(GlobalPooling2D));
export { GlobalMaxPooling2D };
serialization.SerializationMap.register(GlobalMaxPooling2D);
//# sourceMappingURL=pooling.js.map