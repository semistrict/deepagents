//! JSON values cross the boundary as plain Python objects.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Number, Value};

use crate::errors::to_py;

/// A JSON value on its way to Python.
pub struct Json(pub Value);

impl Json {
    pub fn of<T: Serialize>(value: &T) -> PyResult<Json> {
        serde_json::to_value(value).map(Json).map_err(|error| to_py(error.into()))
    }
}

/// Python's `json.loads`, which builds objects faster than converting value by value.
static LOADS: pyo3::sync::PyOnceLock<Py<PyAny>> = pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for Json {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.0 {
            Value::Object(_) | Value::Array(_) => {
                let loads = LOADS.get_or_try_init(py, || py.import("json")?.getattr("loads").map(Bound::unbind))?;
                let text = serde_json::to_string(&self.0).map_err(|error| to_py(error.into()))?;
                loads.bind(py).call1((text,))
            }
            scalar => to_object(py, scalar),
        }
    }
}

/// JSON text on its way to Python, decoded there by `json.loads`.
pub struct JsonText(pub String);

impl<'py> IntoPyObject<'py> for JsonText {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let loads = LOADS.get_or_try_init(py, || py.import("json")?.getattr("loads").map(Bound::unbind))?;
        loads.bind(py).call1((self.0,))
    }
}

fn to_object<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyAny>> {
    Ok(match value {
        Value::Null => py.None().into_bound(py),
        Value::Bool(flag) => PyBool::new(py, *flag).to_owned().into_any(),
        Value::Number(number) => match number.as_i64() {
            Some(int) => int.into_pyobject(py)?.into_any(),
            None => match number.as_u64() {
                Some(int) => int.into_pyobject(py)?.into_any(),
                None => number.as_f64().unwrap_or(f64::NAN).into_pyobject(py)?.into_any(),
            },
        },
        Value::String(text) => PyString::new(py, text).into_any(),
        Value::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(to_object(py, item)?)?;
            }
            list.into_any()
        }
        Value::Object(map) => {
            let dict = PyDict::new(py);
            for (key, item) in map {
                dict.set_item(key, to_object(py, item)?)?;
            }
            dict.into_any()
        }
    })
}

/// Convert a Python object built from dicts, lists, strings, numbers, booleans, and `None`.
pub fn from_object(object: &Bound<'_, PyAny>) -> PyResult<Value> {
    if object.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(flag) = object.cast::<PyBool>() {
        return Ok(Value::Bool(flag.is_true()));
    }
    if object.is_instance_of::<PyInt>() {
        return match object.extract::<i64>() {
            Ok(int) => Ok(Value::from(int)),
            Err(_) => Ok(Value::from(object.extract::<u64>()?)),
        };
    }
    if let Ok(float) = object.cast::<PyFloat>() {
        let value = float.value();
        return Number::from_f64(value).map(Value::Number).ok_or_else(|| PyTypeError::new_err(format!("{value} is not valid JSON")));
    }
    if let Ok(text) = object.cast::<PyString>() {
        return Ok(Value::String(text.to_str()?.to_owned()));
    }
    if let Ok(dict) = object.cast::<PyDict>() {
        let mut map = Map::new();
        for (key, item) in dict.iter() {
            let key = key.cast::<PyString>().map_err(|_| PyTypeError::new_err("JSON object keys must be strings"))?;
            map.insert(key.to_str()?.to_owned(), from_object(&item)?);
        }
        return Ok(Value::Object(map));
    }
    if let Ok(list) = object.cast::<PyList>() {
        return list.iter().map(|item| from_object(&item)).collect::<PyResult<_>>().map(Value::Array);
    }
    if let Ok(tuple) = object.cast::<PyTuple>() {
        return tuple.iter().map(|item| from_object(&item)).collect::<PyResult<_>>().map(Value::Array);
    }
    Err(PyTypeError::new_err(format!("{} is not JSON", object.get_type().name()?)))
}

/// Decode a Python JSON object into a typed record.
pub fn decode<T: DeserializeOwned>(object: &Bound<'_, PyAny>) -> PyResult<T> {
    serde_json::from_value(from_object(object)?).map_err(|error| PyTypeError::new_err(error.to_string()))
}

/// Decode a Python dict into a JSON object map.
pub fn object(object: Option<&Bound<'_, PyAny>>) -> PyResult<Map<String, Value>> {
    match object.map(from_object).transpose()? {
        None | Some(Value::Null) => Ok(Map::new()),
        Some(Value::Object(map)) => Ok(map),
        Some(_) => Err(PyTypeError::new_err("expected a dict")),
    }
}
