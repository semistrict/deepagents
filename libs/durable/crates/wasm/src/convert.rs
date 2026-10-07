//! Records cross the boundary as JSON: serialized by serde here, parsed by
//! the host's native `JSON.parse`, which builds objects faster than
//! converting them value by value.

use js_sys::JSON;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value};
use wasm_bindgen::prelude::*;

use crate::errors::type_error;

/// A Rust value as a JavaScript one.
pub fn to_js<T: Serialize + ?Sized>(value: &T) -> Result<JsValue, JsValue> {
    let text = serde_json::to_string(value).map_err(|error| type_error(&error.to_string()))?;
    JSON::parse(&text)
}

/// JSON text as a JavaScript value.
pub fn parse(text: &str) -> Result<JsValue, JsValue> {
    JSON::parse(text)
}

/// A JavaScript value built from objects, arrays, strings, numbers, booleans, and `null`.
/// `undefined` reads as `null`.
pub fn value(value: &JsValue) -> Result<Value, JsValue> {
    if value.is_undefined() || value.is_null() {
        return Ok(Value::Null);
    }
    let text = JSON::stringify(value)?.as_string().ok_or_else(|| type_error("value is not JSON"))?;
    serde_json::from_str(&text).map_err(|error| type_error(&error.to_string()))
}

/// A JavaScript value decoded into a typed record.
pub fn decode<T: DeserializeOwned>(raw: &JsValue) -> Result<T, JsValue> {
    serde_json::from_value(value(raw)?).map_err(|error| type_error(&error.to_string()))
}

/// A JavaScript object as a JSON map; `null` and `undefined` read as an empty one.
pub fn object(raw: &JsValue) -> Result<Map<String, Value>, JsValue> {
    match value(raw)? {
        Value::Null => Ok(Map::new()),
        Value::Object(map) => Ok(map),
        _ => Err(type_error("expected an object")),
    }
}
