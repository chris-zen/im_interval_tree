/*!
 * An immutable data structure for storing and querying a collection of intervals
 *
 * ```
 * use std::ops::Bound::*;
 * use im_interval_tree::{IntervalTree, Interval};
 *
 * // Construct a tree of intervals
 * let tree : IntervalTree<u8> = IntervalTree::new();
 * let tree = tree.insert(Interval::new(Included(1), Excluded(3)));
 * let tree = tree.insert(Interval::new(Included(2), Excluded(4)));
 * let tree = tree.insert(Interval::new(Included(5), Unbounded));
 * let tree = tree.insert(Interval::new(Excluded(7), Included(8)));
 *
 * // Query for overlapping intervals
 * let query = tree.query_interval(&Interval::new(Included(3), Included(6)));
 * assert_eq!(
 *     query.collect::<Vec<Interval<u8>>>(),
 *     vec![
 *         Interval::new(Included(2), Excluded(4)),
 *         Interval::new(Included(5), Unbounded)
 *     ]
 * );
 *
 * // Query for a specific point
 * let query = tree.query_point(&2);
 * assert_eq!(
 *     query.collect::<Vec<Interval<u8>>>(),
 *     vec![
 *         Interval::new(Included(2), Excluded(4)),
 *         Interval::new(Included(1), Excluded(3))
 *     ]
 * );
 * ```
*/
#[cfg(test)]
mod test;

use std::cmp::*;
use std::ops::Bound;
use std::ops::Bound::*;

mod interval;
mod shared;

pub use crate::interval::Interval;
use crate::interval::*;
use crate::shared::Shared;

#[derive(Clone, Hash)]
struct Node<B: Ord + Clone, D: ToInterval<B> + Clone> {
    data: D,
    left: Option<Shared<Node<B, D>>>,
    right: Option<Shared<Node<B, D>>>,
    height: usize,
    max: Shared<Bound<B>>,
    min: Shared<Bound<B>>,
}

impl<B: Ord + Clone, D: ToInterval<B> + Clone> Node<B, D> {
    fn new(
        data: D,
        left: Option<Shared<Node<B, D>>>,
        right: Option<Shared<Node<B, D>>>,
    ) -> Node<B, D> {
        let height = usize::max(Self::height(&left), Self::height(&right)) + 1;
        let interval = data.to_interval();
        let max = Self::get_max(&interval, &left, &right);
        let min = Self::get_min(&interval, &left, &right);
        Node {
            data,
            left,
            right,
            height,
            max,
            min,
        }
    }

    fn leaf(data: D) -> Node<B, D> {
        Node::new(data, None, None)
    }

    fn height(node: &Option<Shared<Node<B, D>>>) -> usize {
        node.as_ref().map_or(0, |n| n.height)
    }

    fn get_max(
        interval: &Interval<B>,
        left: &Option<Shared<Node<B, D>>>,
        right: &Option<Shared<Node<B, D>>>,
    ) -> Shared<Bound<B>> {
        let mid = &interval.high;
        match (left, right) {
            (None, None) => mid.clone(),
            (None, Some(r)) => high_bound_max(mid, &r.max),
            (Some(l), None) => high_bound_max(mid, &l.max),
            (Some(l), Some(r)) => high_bound_max(mid, &high_bound_max(&l.max, &r.max)),
        }
    }

    fn get_min(
        interval: &Interval<B>,
        left: &Option<Shared<Node<B, D>>>,
        right: &Option<Shared<Node<B, D>>>,
    ) -> Shared<Bound<B>> {
        let mid = &interval.low;
        match (left, right) {
            (None, None) => mid.clone(),
            (None, Some(r)) => low_bound_min(mid, &r.min),
            (Some(l), None) => low_bound_min(mid, &l.min),
            (Some(l), Some(r)) => low_bound_min(mid, &low_bound_min(&l.min, &r.min)),
        }
    }

    fn balance_factor(&self) -> isize {
        (Self::height(&self.left) as isize) - (Self::height(&self.right) as isize)
    }

    fn insert(&self, data: D) -> Self {
        let ordering = data.to_interval().cmp(&self.data.to_interval());
        let res = match ordering {
            Ordering::Less => {
                let insert_left = match &self.left {
                    None => Node::leaf(data),
                    Some(left_tree) => left_tree.insert(data),
                };
                Node::new(
                    self.data.clone(),
                    Some(Shared::new(insert_left)),
                    self.right.clone(),
                )
            }
            Ordering::Greater => {
                let insert_right = match &self.right {
                    None => Node::leaf(data),
                    Some(right_tree) => right_tree.insert(data),
                };
                Node::new(
                    self.data.clone(),
                    self.left.clone(),
                    Some(Shared::new(insert_right)),
                )
            }
            Ordering::Equal => self.clone(),
        };
        res.balance()
    }

    fn get_minimum(&self) -> D {
        match &self.left {
            None => self.data.clone(),
            Some(left_tree) => left_tree.get_minimum(),
        }
    }

    fn remove(&self, data: &D) -> Option<Shared<Self>> {
        let ordering = data.to_interval().cmp(&self.data.to_interval());
        let res = match ordering {
            Ordering::Equal => match (&self.left, &self.right) {
                (None, None) => None,
                (Some(left_tree), None) => Some(left_tree.clone()),
                (None, Some(right_tree)) => Some(right_tree.clone()),
                (Some(_), Some(right_tree)) => {
                    let successor = right_tree.get_minimum();
                    let new_right = right_tree.remove(&successor);
                    let new_node = Node::new(successor, self.left.clone(), new_right);
                    Some(Shared::new(new_node))
                }
            },
            Ordering::Less => match &self.left {
                None => Some(Shared::new(self.clone())),
                Some(left_tree) => Some(Shared::new(self.replace_left(left_tree.remove(data)))),
            },
            Ordering::Greater => match &self.right {
                None => Some(Shared::new(self.clone())),
                Some(right_tree) => Some(Shared::new(self.replace_right(right_tree.remove(data)))),
            },
        };
        res.map(|r| Shared::new(r.balance()))
    }

    fn replace_left(&self, new_left: Option<Shared<Node<B, D>>>) -> Node<B, D> {
        Self::new(self.data.clone(), new_left, self.right.clone())
    }

    fn replace_right(&self, new_right: Option<Shared<Node<B, D>>>) -> Node<B, D> {
        Self::new(self.data.clone(), self.left.clone(), new_right)
    }

    fn rotate_right(&self) -> Self {
        let pivot = self.left.as_ref().unwrap();
        let new_right = self.replace_left(pivot.right.clone());
        pivot.replace_right(Some(Shared::new(new_right)))
    }

    fn rotate_left(&self) -> Self {
        let pivot = self.right.as_ref().unwrap();
        let new_left = self.replace_right(pivot.left.clone());
        pivot.replace_left(Some(Shared::new(new_left)))
    }

    fn balance(&self) -> Self {
        let balance_factor = self.balance_factor();
        if balance_factor < -1 {
            let right = self.right.as_ref().unwrap();
            if right.balance_factor() > 0 {
                self.replace_right(Some(Shared::new(right.rotate_right())))
                    .rotate_left()
            } else {
                self.rotate_left()
            }
        } else if balance_factor > 1 {
            let left = self.left.as_ref().unwrap();
            if left.balance_factor() < 0 {
                self.replace_left(Some(Shared::new(left.rotate_left())))
                    .rotate_right()
            } else {
                self.rotate_right()
            }
        } else {
            self.clone()
        }
    }
}

/// An Iterator over Intervals matching some query
pub struct Iter<B: Ord + Clone, D: ToInterval<B> + Clone> {
    stack: Vec<Shared<Node<B, D>>>,
    query: Interval<B>,
}

impl<B: Ord + Clone, D: ToInterval<B> + Clone> Iterator for Iter<B, D> {
    type Item = D;
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(node) = self.stack.pop() {
            if let Some(left_tree) = &node.left {
                let max_is_gte = match (&*left_tree.max, self.query.low()) {
                    (Included(max), Included(low)) => max >= low,
                    (Included(max), Excluded(low))
                    | (Excluded(max), Included(low))
                    | (Excluded(max), Excluded(low)) => max > low,
                    _ => true,
                };
                if max_is_gte {
                    self.stack.push(left_tree.clone())
                }
            }
            if let Some(right_tree) = &node.right {
                let min_is_lte = match (&*right_tree.min, self.query.high()) {
                    (Included(min), Included(high)) => min <= high,
                    (Included(min), Excluded(high))
                    | (Excluded(min), Included(high))
                    | (Excluded(min), Excluded(high)) => min < high,
                    _ => true,
                };
                if min_is_lte {
                    self.stack.push(right_tree.clone())
                }
            }
            if self.query.overlaps(&node.data.to_interval()) {
                return Some(node.data.clone());
            }
        }
        None
    }
}

/// An immutable data structure for storing and querying a collection of intervals
///
/// # Example
/// ```
/// use std::ops::Bound::*;
/// use im_interval_tree::{IntervalTree, Interval};
///
/// // Construct a tree of intervals
/// let tree : IntervalTree<u8> = IntervalTree::new();
/// let tree = tree.insert(Interval::new(Included(1), Excluded(3)));
/// let tree = tree.insert(Interval::new(Included(2), Excluded(4)));
/// let tree = tree.insert(Interval::new(Included(5), Unbounded));
/// let tree = tree.insert(Interval::new(Excluded(7), Included(8)));
///
/// // Query for overlapping intervals
/// let query = tree.query_interval(&Interval::new(Included(3), Included(6)));
/// assert_eq!(
///     query.collect::<Vec<Interval<u8>>>(),
///     vec![
///         Interval::new(Included(2), Excluded(4)),
///         Interval::new(Included(5), Unbounded)
///     ]
/// );
///
/// // Query for a specific point
/// let query = tree.query_point(&2);
/// assert_eq!(
///     query.collect::<Vec<Interval<u8>>>(),
///     vec![
///         Interval::new(Included(2), Excluded(4)),
///         Interval::new(Included(1), Excluded(3))
///     ]
/// );
/// ```
#[derive(Clone, Hash)]
pub struct IntervalTree<B, D = Interval<B>>
where
    B: Ord + Clone,
    D: ToInterval<B> + Clone,
{
    root: Option<Shared<Node<B, D>>>,
}

impl<B: Ord + Clone, D: ToInterval<B> + Clone> IntervalTree<B, D> {
    /// Construct an empty IntervalTree
    pub fn new() -> IntervalTree<B, D> {
        IntervalTree { root: None }
    }

    /// Construct a new IntervalTree with the given Interval added
    ///
    /// # Example
    /// ```
    /// # use std::ops::Bound::*;
    /// # use im_interval_tree::{IntervalTree, Interval};
    /// let tree : IntervalTree<u8> = IntervalTree::new();
    /// let tree = tree.insert(Interval::new(Included(1), Included(2)));
    /// assert_eq!(
    ///     tree.iter().collect::<Vec<Interval<u8>>>(),
    ///     vec![Interval::new(Included(1), Included(2))]
    /// );
    /// ```
    pub fn insert(&self, data: D) -> IntervalTree<B, D> {
        let new_root = match &self.root {
            None => Node::leaf(data),
            Some(node) => node.insert(data),
        };
        IntervalTree {
            root: Some(Shared::new(new_root)),
        }
    }

    /// Construct a new IntervalTree minus the given Interval, if present
    ///
    /// # Example
    /// ```
    /// # use std::ops::Bound::*;
    /// # use im_interval_tree::{IntervalTree, Interval};
    /// let tree : IntervalTree<u8> = IntervalTree::new();
    /// let tree = tree.insert(Interval::new(Included(1), Included(2)));
    /// let tree = tree.insert(Interval::new(Included(1), Included(3)));
    ///
    /// let tree = tree.remove(&Interval::new(Included(1), Included(2)));
    /// assert_eq!(
    ///     tree.iter().collect::<Vec<Interval<u8>>>(),
    ///     vec![Interval::new(Included(1), Included(3))]
    /// );
    /// ```
    pub fn remove(&self, data: &D) -> IntervalTree<B, D> {
        match &self.root {
            None => IntervalTree::new(),
            Some(node) => IntervalTree {
                root: node.remove(data),
            },
        }
    }

    /// Return an Iterator over all the intervals in the tree that overlap
    /// with the given interval
    ///
    /// # Example
    /// ```
    /// # use std::ops::Bound::*;
    /// # use im_interval_tree::{IntervalTree, Interval};
    /// let tree : IntervalTree<u8> = IntervalTree::new();
    /// let tree = tree.insert(Interval::new(Included(1), Excluded(3)));
    /// let tree = tree.insert(Interval::new(Included(5), Unbounded));
    ///
    /// let query = tree.query_interval(&Interval::new(Included(3), Included(6)));
    /// assert_eq!(
    ///     query.collect::<Vec<Interval<u8>>>(),
    ///     vec![Interval::new(Included(5), Unbounded)]
    /// );
    /// ```
    pub fn query_interval(&self, interval: &Interval<B>) -> impl Iterator<Item = D> + '_ {
        let mut stack = Vec::new();
        if let Some(node) = &self.root {
            stack.push(node.clone())
        }
        Iter {
            stack,
            query: interval.clone(),
        }
    }

    /// Return an Iterator over all the intervals in the tree that contain
    /// the given point
    ///
    /// This is equivalent to `tree.query_interval(Interval::new(Included(point), Included(point)))`
    ///
    /// # Example
    /// ```
    /// # use std::ops::Bound::*;
    /// # use im_interval_tree::{IntervalTree, Interval};
    /// let tree : IntervalTree<u8> = IntervalTree::new();
    /// let tree = tree.insert(Interval::new(Included(1), Excluded(3)));
    /// let tree = tree.insert(Interval::new(Included(5), Unbounded));
    ///
    /// let query = tree.query_point(&2);
    /// assert_eq!(
    ///     query.collect::<Vec<Interval<u8>>>(),
    ///     vec![Interval::new(Included(1), Excluded(3))]
    /// );
    /// ```
    pub fn query_point(&self, point: &B) -> impl Iterator<Item = D> + '_ {
        let interval = Interval::new(Included(point.clone()), Included(point.clone()));
        self.query_interval(&interval)
    }

    /// Return an Iterator over all the intervals in the tree
    ///
    /// This is equivalent to `tree.query_interval(Unbounded, Unbounded)`
    ///
    /// # Example
    /// ```
    /// # use std::ops::Bound::*;
    /// # use im_interval_tree::{IntervalTree, Interval};
    /// let tree : IntervalTree<u8> = IntervalTree::new();
    /// let tree = tree.insert(Interval::new(Included(2), Excluded(4)));
    /// let tree = tree.insert(Interval::new(Included(5), Unbounded));
    ///
    /// let iter = tree.iter();
    /// assert_eq!(
    ///     iter.collect::<Vec<Interval<u8>>>(),
    ///     vec![
    ///         Interval::new(Included(2), Excluded(4)),
    ///         Interval::new(Included(5), Unbounded),
    ///     ]
    /// );
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = D> + '_ {
        self.query_interval(&Interval::new(Unbounded, Unbounded))
    }
}

impl<B: Ord + Clone, D: ToInterval<B> + Clone> Default for IntervalTree<B, D> {
    fn default() -> Self {
        Self::new()
    }
}
