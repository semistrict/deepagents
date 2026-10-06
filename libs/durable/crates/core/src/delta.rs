//! Chord delta operations over plain JSON.
//!
//! Documents are stored as a complete base plus batches of Chord ops, the
//! tuples pi-durable writes (`@earendil-works/chord/delta`). [`apply`] accepts
//! every verb. [`diff`] derives a valid batch between two values, so callers
//! only ever assign whole values; appends to strings and arrays stay
//! proportional to the change. Op shape is not canonical: only the resulting
//! value is.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::error::{Error, Result};

/// One step into a JSON value: an object key or an array index.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Segment {
    Index(usize),
    Key(String),
}

pub type Path = Vec<Segment>;

/// One Chord operation. `Set`, `Delete`, `Append`, and `Trim` never target the root.
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    /// `["r", value]`: replace the whole value.
    Replace(Value),
    /// `["s", path, value]`: write an object member or array element.
    Set(Path, Value),
    /// `["d", path]`: delete an object member, or remove an array element.
    Delete(Path),
    /// `["a", path, text]`: append to a string.
    Append(Path, String),
    /// `["t", path, count]`: drop the first `count` UTF-16 units of a string.
    Trim(Path, usize),
    /// `["p", path, index, remove, items]`: splice an array.
    Splice(Path, usize, usize, Vec<Value>),
    /// `["m", path, permutation]`: reorder an array, `new[i] = old[permutation[i]]`.
    Permute(Path, Vec<usize>),
}

impl Serialize for Op {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let path = |path: &Path| serde_json::to_value(path).expect("paths serialize");
        let tuple = match self {
            Op::Replace(value) => json!(["r", value]),
            Op::Set(p, value) => json!(["s", path(p), value]),
            Op::Delete(p) => json!(["d", path(p)]),
            Op::Append(p, text) => json!(["a", path(p), text]),
            Op::Trim(p, count) => json!(["t", path(p), count]),
            Op::Splice(p, index, remove, items) => json!(["p", path(p), index, remove, items]),
            Op::Permute(p, permutation) => json!(["m", path(p), permutation]),
        };
        tuple.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Op {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        Op::decode(value).map_err(serde::de::Error::custom)
    }
}

impl Op {
    fn decode(value: Value) -> std::result::Result<Op, String> {
        let Value::Array(mut parts) = value else {
            return Err("op is not a tuple".into());
        };
        let verb = parts.first().and_then(Value::as_str).ok_or("op has no verb")?.to_owned();
        let arity = |n: usize| {
            if parts.len() == n { Ok(()) } else { Err(format!("{verb} arity")) }
        };
        let count = |value: &Value| value.as_u64().map(|n| n as usize).ok_or_else(|| format!("{verb} count"));
        let path = |value: Value| serde_json::from_value::<Path>(value).map_err(|error| error.to_string());
        let op = match verb.as_str() {
            "r" => {
                arity(2)?;
                Op::Replace(parts.pop().expect("arity checked"))
            }
            "s" => {
                arity(3)?;
                let value = parts.pop().expect("arity checked");
                Op::Set(path(parts.pop().expect("arity checked"))?, value)
            }
            "d" => {
                arity(2)?;
                Op::Delete(path(parts.pop().expect("arity checked"))?)
            }
            "a" => {
                arity(3)?;
                let Some(Value::String(text)) = parts.pop() else {
                    return Err("a value".into());
                };
                Op::Append(path(parts.pop().expect("arity checked"))?, text)
            }
            "t" => {
                arity(3)?;
                let trimmed = count(&parts[2])?;
                Op::Trim(path(parts.swap_remove(1))?, trimmed)
            }
            "p" => {
                arity(5)?;
                let Some(Value::Array(items)) = parts.pop() else {
                    return Err("p items".into());
                };
                let (index, remove) = (count(&parts[2])?, count(&parts[3])?);
                Op::Splice(path(parts.swap_remove(1))?, index, remove, items)
            }
            "m" => {
                arity(3)?;
                let permutation = serde_json::from_value(parts.pop().expect("arity checked")).map_err(|error| error.to_string())?;
                Op::Permute(path(parts.pop().expect("arity checked"))?, permutation)
            }
            other => return Err(format!("unknown op verb: {other}")),
        };
        Ok(op)
    }
}

/// The operations that turn `old` into `new`.
pub fn diff(old: &Value, new: &Value) -> Vec<Op> {
    let mut ops = Vec::new();
    if old != new {
        if compatible(old, new) {
            diff_into(&mut Vec::new(), old, new, &mut ops);
        } else {
            ops.push(Op::Replace(new.clone()));
        }
    }
    ops
}

/// Whether `old` can be edited into `new` in place at the root.
fn compatible(old: &Value, new: &Value) -> bool {
    matches!((old, new), (Value::Object(_), Value::Object(_)) | (Value::Array(_), Value::Array(_)))
}

fn diff_into(path: &mut Path, old: &Value, new: &Value, ops: &mut Vec<Op>) {
    match (old, new) {
        (Value::Object(before), Value::Object(after)) => {
            for key in before.keys().filter(|key| !after.contains_key(*key)) {
                ops.push(Op::Delete(child(path, Segment::Key(key.clone()))));
            }
            for (key, value) in after {
                path.push(Segment::Key(key.clone()));
                match before.get(key) {
                    Some(previous) if previous == value => {}
                    Some(previous) => edit(path, previous, value, ops),
                    None => ops.push(Op::Set(path.clone(), value.clone())),
                }
                path.pop();
            }
        }
        (Value::Array(before), Value::Array(after)) => {
            let shared = before.len().min(after.len());
            let start = ops.len();
            for index in 0..shared {
                if before[index] != after[index] {
                    path.push(Segment::Index(index));
                    edit(path, &before[index], &after[index], ops);
                    path.pop();
                }
            }
            // Rewriting most items costs more than one replacement.
            if ops.len() - start > shared / 2 + 1 {
                ops.truncate(start);
                ops.push(Op::Splice(path.clone(), 0, before.len(), after.clone()));
                return;
            }
            if after.len() > shared {
                ops.push(Op::Splice(path.clone(), shared, 0, after[shared..].to_vec()));
            } else if before.len() > shared {
                ops.push(Op::Splice(path.clone(), shared, before.len() - shared, Vec::new()));
            }
        }
        _ => unreachable!("diff_into only descends into matching containers"),
    }
}

/// Change the non-root value at `path` from `old` to `new`.
fn edit(path: &Path, old: &Value, new: &Value, ops: &mut Vec<Op>) {
    match (old, new) {
        (Value::String(before), Value::String(after)) if after.starts_with(before.as_str()) => {
            ops.push(Op::Append(path.clone(), after[before.len()..].to_owned()));
        }
        _ if compatible(old, new) => diff_into(&mut path.clone(), old, new, ops),
        _ => ops.push(Op::Set(path.clone(), new.clone())),
    }
}

fn child(path: &Path, segment: Segment) -> Path {
    let mut path = path.clone();
    path.push(segment);
    path
}

/// Apply `ops` to `value` in order.
pub fn apply(value: &mut Value, ops: &[Op]) -> Result<()> {
    for op in ops {
        apply_one(value, op)?;
    }
    Ok(())
}

fn apply_one(root: &mut Value, op: &Op) -> Result<()> {
    match op {
        Op::Replace(value) => *root = value.clone(),
        Op::Splice(path, index, remove, items) => {
            let Value::Array(target) = resolve(root, path)? else {
                return Err(unresolvable(op));
            };
            if index + remove > target.len() {
                return Err(unresolvable(op));
            }
            target.splice(*index..*index + *remove, items.iter().cloned());
        }
        Op::Permute(path, permutation) => {
            let Value::Array(target) = resolve(root, path)? else {
                return Err(unresolvable(op));
            };
            if target.len() != permutation.len() || permutation.iter().any(|&from| from >= target.len()) {
                return Err(unresolvable(op));
            }
            let previous = std::mem::take(target);
            *target = permutation.iter().map(|&from| previous[from].clone()).collect();
        }
        Op::Set(path, _) | Op::Delete(path) | Op::Append(path, _) | Op::Trim(path, _) => {
            let (last, parent_path) = path.split_last().ok_or_else(|| unresolvable(op))?;
            let parent = resolve(root, parent_path)?;
            match (op, parent, last) {
                (Op::Set(_, value), Value::Object(map), Segment::Key(key)) => {
                    map.insert(key.clone(), value.clone());
                }
                (Op::Set(_, value), Value::Array(items), Segment::Index(index)) if *index <= items.len() => {
                    if *index == items.len() {
                        items.push(value.clone());
                    } else {
                        items[*index] = value.clone();
                    }
                }
                (Op::Delete(_), Value::Object(map), Segment::Key(key)) => {
                    map.remove(key);
                }
                (Op::Delete(_), Value::Array(items), Segment::Index(index)) if *index < items.len() => {
                    items.remove(*index);
                }
                (Op::Append(_, text), parent, last) => match member(parent, last) {
                    Some(Value::String(current)) => current.push_str(text),
                    _ => return Err(unresolvable(op)),
                },
                (Op::Trim(_, count), parent, last) => match member(parent, last) {
                    Some(Value::String(current)) => *current = trim_utf16(current, *count),
                    _ => return Err(unresolvable(op)),
                },
                _ => return Err(unresolvable(op)),
            }
        }
    }
    Ok(())
}

fn member<'a>(parent: &'a mut Value, segment: &Segment) -> Option<&'a mut Value> {
    match (parent, segment) {
        (Value::Object(map), Segment::Key(key)) => map.get_mut(key),
        (Value::Array(items), Segment::Index(index)) => items.get_mut(*index),
        _ => None,
    }
}

/// JavaScript's `string.slice(count)`: offsets count UTF-16 code units.
fn trim_utf16(text: &str, count: usize) -> String {
    let units: Vec<u16> = text.encode_utf16().skip(count).collect();
    String::from_utf16_lossy(&units)
}

fn resolve<'a>(mut value: &'a mut Value, path: &[Segment]) -> Result<&'a mut Value> {
    for segment in path {
        value = member(value, segment).ok_or_else(|| Error::Corrupt(format!("unresolvable path: {path:?}")))?;
    }
    Ok(value)
}

fn unresolvable(op: &Op) -> Error {
    Error::Corrupt(format!("operation cannot apply: {}", serde_json::to_string(op).unwrap_or_default()))
}
