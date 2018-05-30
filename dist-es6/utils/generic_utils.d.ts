import { DataType, serialization, Tensor } from '@tensorflow/tfjs-core';
import { Shape } from '../types';
export declare function pyListRepeat(value: any, numValues: number): any[];
export declare function assert(val: boolean, message?: string): void;
export declare function count<T>(array: T[], refernce: T): number;
export declare function singletonOrArray<T>(xs: T[]): T | T[];
export declare function toList(x: any): any[];
export declare function objectListUid(objs: any | any[]): string;
export declare function isArrayOfShapes(x: Shape | Shape[]): boolean;
export declare function normalizeShapeList(x: Shape | Shape[]): Shape[];
export declare function toSnakeCase(name: string): string;
export declare function toCamelCase(identifier: string): string;
export declare function serializeKerasObject(instance: serialization.Serializable): serialization.ConfigDictValue;
export declare function deserializeKerasObject(identifier: string | serialization.ConfigDict, moduleObjects?: {
    [objName: string]: any;
}, customObjects?: {
    [objName: string]: any;
}, printableModuleName?: string): any;
export declare function getExactlyOneTensor(xs: Tensor | Tensor[]): Tensor;
export declare function getExactlyOneShape(shapes: Shape | Shape[]): Shape;
export declare function numberCompare(a: number, b: number): 0 | 1 | -1;
export declare function reverseNumberCompare(a: number, b: number): number;
export declare function stringToDType(dtype: string): DataType;
export declare function stringsEqual(xs: string[], ys: string[]): boolean;
export declare function unique<T>(xs: T[]): T[];
export declare function isObjectEmpty(obj: {}): boolean;
export declare function checkStringTypeUnionValue(values: string[], label: string, value: string): void;
