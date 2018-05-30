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
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
import { doc, eye, linalg, ones, randomUniform, scalar, serialization, tidy, truncatedNormal, zeros } from '@tensorflow/tfjs-core';
import * as K from './backend/tfjs_backend';
import { checkDataFormat } from './common';
import { NotImplementedError, ValueError } from './errors';
import { checkStringTypeUnionValue, deserializeKerasObject, serializeKerasObject } from './utils/generic_utils';
import { arrayProd } from './utils/math_utils';
export var VALID_FAN_MODE_VALUES = ['fanIn', 'fanOut', 'fanAvg'];
export function checkFanMode(value) {
    checkStringTypeUnionValue(VALID_FAN_MODE_VALUES, 'FanMode', value);
}
export var VALID_DISTRIBUTION_VALUES = ['normal', 'uniform'];
export function checkDistribution(value) {
    checkStringTypeUnionValue(VALID_DISTRIBUTION_VALUES, 'Distribution', value);
}
var Initializer = (function (_super) {
    __extends(Initializer, _super);
    function Initializer() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    Initializer.prototype.fromConfigUsesCustomObjects = function () {
        return false;
    };
    Initializer.prototype.getConfig = function () {
        return {};
    };
    Initializer = __decorate([
        doc({ heading: 'Initializers', subheading: 'Classes', namespace: 'initializers' })
    ], Initializer);
    return Initializer;
}(serialization.Serializable));
export { Initializer };
var Zeros = (function (_super) {
    __extends(Zeros, _super);
    function Zeros() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    Zeros.prototype.apply = function (shape, dtype) {
        return zeros(shape, dtype);
    };
    Zeros.className = 'Zeros';
    return Zeros;
}(Initializer));
export { Zeros };
serialization.SerializationMap.register(Zeros);
var Ones = (function (_super) {
    __extends(Ones, _super);
    function Ones() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    Ones.prototype.apply = function (shape, dtype) {
        return ones(shape, dtype);
    };
    Ones.className = 'Ones';
    return Ones;
}(Initializer));
export { Ones };
serialization.SerializationMap.register(Ones);
var Constant = (function (_super) {
    __extends(Constant, _super);
    function Constant(config) {
        var _this = _super.call(this) || this;
        _this.value = config.value;
        return _this;
    }
    Constant.prototype.apply = function (shape, dtype) {
        var _this = this;
        return tidy(function () { return K.scalarTimesArray(scalar(_this.value), ones(shape, dtype)); });
    };
    Constant.prototype.getConfig = function () {
        return {
            value: this.value,
        };
    };
    Constant.className = 'Constant';
    return Constant;
}(Initializer));
export { Constant };
serialization.SerializationMap.register(Constant);
var RandomUniform = (function (_super) {
    __extends(RandomUniform, _super);
    function RandomUniform(config) {
        var _this = _super.call(this) || this;
        _this.DEFAULT_MINVAL = -0.05;
        _this.DEFAULT_MAXVAL = 0.05;
        _this.minval = config.minval || _this.DEFAULT_MINVAL;
        _this.maxval = config.maxval || _this.DEFAULT_MAXVAL;
        _this.seed = config.seed;
        return _this;
    }
    RandomUniform.prototype.apply = function (shape, dtype) {
        return randomUniform(shape, this.minval, this.maxval, dtype);
    };
    RandomUniform.prototype.getConfig = function () {
        return { minval: this.minval, maxval: this.maxval, seed: this.seed };
    };
    RandomUniform.className = 'RandomUniform';
    return RandomUniform;
}(Initializer));
export { RandomUniform };
serialization.SerializationMap.register(RandomUniform);
var RandomNormal = (function (_super) {
    __extends(RandomNormal, _super);
    function RandomNormal(config) {
        var _this = _super.call(this) || this;
        _this.DEFAULT_MEAN = 0.;
        _this.DEFAULT_STDDEV = 0.05;
        _this.mean = config.mean || _this.DEFAULT_MEAN;
        _this.stddev = config.stddev || _this.DEFAULT_STDDEV;
        _this.seed = config.seed;
        return _this;
    }
    RandomNormal.prototype.apply = function (shape, dtype) {
        if (dtype === 'bool') {
            throw new NotImplementedError("randomNormal does not support dType bool.");
        }
        return K.randomNormal(shape, this.mean, this.stddev, dtype, this.seed);
    };
    RandomNormal.prototype.getConfig = function () {
        return { mean: this.mean, stddev: this.stddev, seed: this.seed };
    };
    RandomNormal.className = 'RandomNormal';
    return RandomNormal;
}(Initializer));
export { RandomNormal };
serialization.SerializationMap.register(RandomNormal);
var TruncatedNormal = (function (_super) {
    __extends(TruncatedNormal, _super);
    function TruncatedNormal(config) {
        var _this = _super.call(this) || this;
        _this.DEFAULT_MEAN = 0.;
        _this.DEFAULT_STDDEV = 0.05;
        _this.mean = config.mean || _this.DEFAULT_MEAN;
        _this.stddev = config.stddev || _this.DEFAULT_STDDEV;
        _this.seed = config.seed;
        return _this;
    }
    TruncatedNormal.prototype.apply = function (shape, dtype) {
        if (dtype === 'bool') {
            throw new NotImplementedError("truncatedNormal does not support dType bool.");
        }
        return truncatedNormal(shape, this.mean, this.stddev, dtype, this.seed);
    };
    TruncatedNormal.prototype.getConfig = function () {
        return { mean: this.mean, stddev: this.stddev, seed: this.seed };
    };
    TruncatedNormal.className = 'TruncatedNormal';
    return TruncatedNormal;
}(Initializer));
export { TruncatedNormal };
serialization.SerializationMap.register(TruncatedNormal);
var Identity = (function (_super) {
    __extends(Identity, _super);
    function Identity(config) {
        var _this = _super.call(this) || this;
        _this.gain = config.gain != null ? scalar(config.gain) : K.getScalar(1.0);
        return _this;
    }
    Identity.prototype.apply = function (shape, dtype) {
        var _this = this;
        return tidy(function () {
            if (shape.length !== 2 || shape[0] !== shape[1]) {
                throw new ValueError('Identity matrix initializer can only be used for' +
                    ' 2D square matrices.');
            }
            else {
                return K.scalarTimesArray(_this.gain, eye(shape[0]));
            }
        });
    };
    Identity.prototype.getConfig = function () {
        return { gain: this.gain.get() };
    };
    Identity.className = 'Identity';
    return Identity;
}(Initializer));
export { Identity };
serialization.SerializationMap.register(Identity);
function computeFans(shape, dataFormat) {
    if (dataFormat === void 0) { dataFormat = 'channelsLast'; }
    var fanIn;
    var fanOut;
    checkDataFormat(dataFormat);
    if (shape.length === 2) {
        fanIn = shape[0];
        fanOut = shape[1];
    }
    else if ([3, 4, 5].indexOf(shape.length) !== -1) {
        if (dataFormat === 'channelsFirst') {
            var receptiveFieldSize = arrayProd(shape, 2);
            fanIn = shape[1] * receptiveFieldSize;
            fanOut = shape[0] * receptiveFieldSize;
        }
        else if (dataFormat === 'channelsLast') {
            var receptiveFieldSize = arrayProd(shape, 0, shape.length - 2);
            fanIn = shape[shape.length - 2] * receptiveFieldSize;
            fanOut = shape[shape.length - 1] * receptiveFieldSize;
        }
    }
    else {
        var shapeProd = arrayProd(shape);
        fanIn = Math.sqrt(shapeProd);
        fanOut = Math.sqrt(shapeProd);
    }
    return [fanIn, fanOut];
}
var VarianceScaling = (function (_super) {
    __extends(VarianceScaling, _super);
    function VarianceScaling(config) {
        var _this = _super.call(this) || this;
        if (config.scale < 0.0) {
            throw new ValueError("scale must be a positive float. Got: " + config.scale);
        }
        _this.scale = config.scale == null ? 1.0 : config.scale;
        _this.mode = config.mode;
        checkFanMode(_this.mode);
        _this.distribution = config.distribution;
        checkDistribution(_this.distribution);
        _this.seed = config.seed;
        return _this;
    }
    VarianceScaling.prototype.apply = function (shape, dtype) {
        var fans = computeFans(shape);
        var fanIn = fans[0];
        var fanOut = fans[1];
        var scale = this.scale;
        if (this.mode === 'fanIn') {
            scale /= Math.max(1, fanIn);
        }
        else if (this.mode === 'fanOut') {
            scale /= Math.max(1, fanOut);
        }
        else {
            scale /= Math.max(1, (fanIn + fanOut) / 2);
        }
        if (this.distribution === 'normal') {
            var stddev = Math.sqrt(scale);
            if (dtype === 'bool') {
                throw new NotImplementedError(this.getClassName() + " does not support dType bool.");
            }
            return truncatedNormal(shape, 0, stddev, dtype, this.seed);
        }
        else {
            var limit = Math.sqrt(3 * scale);
            return randomUniform(shape, -limit, limit, dtype);
        }
    };
    VarianceScaling.prototype.getConfig = function () {
        return {
            scale: this.scale,
            mode: this.mode,
            distribution: this.distribution,
            seed: this.seed
        };
    };
    VarianceScaling.className = 'VarianceScaling';
    return VarianceScaling;
}(Initializer));
export { VarianceScaling };
serialization.SerializationMap.register(VarianceScaling);
var GlorotUniform = (function (_super) {
    __extends(GlorotUniform, _super);
    function GlorotUniform(config) {
        return _super.call(this, {
            scale: 1.0,
            mode: 'fanAvg',
            distribution: 'uniform',
            seed: config == null ? null : config.seed
        }) || this;
    }
    GlorotUniform.prototype.getClassName = function () {
        return VarianceScaling.className;
    };
    return GlorotUniform;
}(VarianceScaling));
export { GlorotUniform };
var GlorotNormal = (function (_super) {
    __extends(GlorotNormal, _super);
    function GlorotNormal(config) {
        return _super.call(this, {
            scale: 1.0,
            mode: 'fanAvg',
            distribution: 'normal',
            seed: config == null ? null : config.seed
        }) || this;
    }
    GlorotNormal.prototype.getClassName = function () {
        return VarianceScaling.className;
    };
    return GlorotNormal;
}(VarianceScaling));
export { GlorotNormal };
var HeNormal = (function (_super) {
    __extends(HeNormal, _super);
    function HeNormal(config) {
        return _super.call(this, {
            scale: 2.0,
            mode: 'fanIn',
            distribution: 'normal',
            seed: config == null ? null : config.seed
        }) || this;
    }
    HeNormal.prototype.getClassName = function () {
        return VarianceScaling.className;
    };
    return HeNormal;
}(VarianceScaling));
export { HeNormal };
var LeCunNormal = (function (_super) {
    __extends(LeCunNormal, _super);
    function LeCunNormal(config) {
        return _super.call(this, {
            scale: 1.0,
            mode: 'fanIn',
            distribution: 'normal',
            seed: config == null ? null : config.seed
        }) || this;
    }
    LeCunNormal.prototype.getClassName = function () {
        return VarianceScaling.className;
    };
    return LeCunNormal;
}(VarianceScaling));
export { LeCunNormal };
var Orthogonal = (function (_super) {
    __extends(Orthogonal, _super);
    function Orthogonal(config) {
        var _this = _super.call(this) || this;
        _this.DEFAULT_GAIN = 1;
        _this.gain = config.gain == null ? _this.DEFAULT_GAIN : config.gain;
        _this.seed = config.seed;
        if (_this.seed != null) {
            throw new NotImplementedError('Random seed is not implemented for Orthogonal Initializer yet.');
        }
        return _this;
    }
    Orthogonal.prototype.apply = function (shape, dtype) {
        var _this = this;
        return tidy(function () {
            if (shape.length !== 2) {
                throw new NotImplementedError('The Orthogonal Initializer does not support non-2D shapes yet.');
            }
            if (shape[0] * shape[1] > 2000) {
                console.warn("Orthgonal initializer is being called on a matrix with more " +
                    (" than 2000 (" + shape[0] * shape[1] + ") elements: ") +
                    "Slowness may result.");
            }
            var normalizedShape = shape[0] > shape[1] ? [shape[1], shape[0]] : shape;
            var a = K.randomNormal(normalizedShape, 0, 1, 'float32');
            var q = linalg.gramSchmidt(a);
            if (shape[0] > shape[1]) {
                q = q.transpose();
            }
            return K.scalarTimesArray(K.getScalar(_this.gain), q);
        });
    };
    Orthogonal.prototype.getConfig = function () {
        return {
            gain: this.gain,
            seed: this.seed,
        };
    };
    Orthogonal.className = 'Orthogonal';
    return Orthogonal;
}(Initializer));
export { Orthogonal };
serialization.SerializationMap.register(Orthogonal);
export var INITIALIZER_IDENTIFIER_REGISTRY_SYMBOL_MAP = {
    'constant': 'Constant',
    'glorotNormal': 'GlorotNormal',
    'glorotUniform': 'GlorotUniform',
    'heNormal': 'HeNormal',
    'identity': 'Identity',
    'leCunNormal': 'LeCunNormal',
    'ones': 'Ones',
    'orthogonal': 'Orthogonal',
    'randomNormal': 'RandomNormal',
    'randomUniform': 'RandomUniform',
    'truncatedNormal': 'TruncatedNormal',
    'varianceScaling': 'VarianceScaling',
    'zeros': 'Zeros'
};
function deserializeInitializer(config, customObjects) {
    if (customObjects === void 0) { customObjects = {}; }
    return deserializeKerasObject(config, serialization.SerializationMap.getMap().classNameMap, customObjects, 'initializer');
}
export function serializeInitializer(initializer) {
    return serializeKerasObject(initializer);
}
export function getInitializer(identifier) {
    if (typeof identifier === 'string') {
        var className = identifier in INITIALIZER_IDENTIFIER_REGISTRY_SYMBOL_MAP ?
            INITIALIZER_IDENTIFIER_REGISTRY_SYMBOL_MAP[identifier] :
            identifier;
        if (className === 'GlorotUniform') {
            return new GlorotUniform();
        }
        else if (className === 'GlorotNormal') {
            return new GlorotNormal();
        }
        else if (className === 'HeNormal') {
            return new HeNormal();
        }
        else if (className === 'LeCunNormal') {
            return new LeCunNormal();
        }
        else {
            var config = { className: className, config: {} };
            return deserializeInitializer(config);
        }
    }
    else if (identifier instanceof Initializer) {
        return identifier;
    }
    else {
        return deserializeInitializer(identifier);
    }
}
//# sourceMappingURL=initializers.js.map