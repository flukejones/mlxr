use mlx_rs::{
    error::Exception,
    fast::{scaled_dot_product_attention, ScaledDotProductAttentionMask},
    ops::{
        indexing::{Ellipsis, IndexOp, TryIndexMutOp},
        zeros_dtype,
    },
    Array,
};

/// Initial KV buffer capacity. Doubles geometrically on overflow.
pub const DEFAULT_KV_CACHE_INIT_CAPACITY: i32 = 64;

/// `None` + `n_q > 1` is causal by definition; `None` + decode needs no mask.
#[inline]
fn resolve_sdpa_mask(mask: Option<&Array>, n_q: i32) -> Option<ScaledDotProductAttentionMask<'_>> {
    match mask {
        Some(m) => Some(ScaledDotProductAttentionMask::Array(m)),
        None if n_q > 1 => Some(ScaledDotProductAttentionMask::Causal),
        None => None,
    }
}

/// Catches a `[L, L]` causal mask built without `cache.offset()`: turn 1
/// passes silently, turn 2 fails inside SDPA with a cryptic broadcast
/// error.
#[inline]
fn assert_mask_matches_keys(mask: Option<&Array>, k_full: &Array) {
    if !cfg!(debug_assertions) {
        return;
    }
    let Some(mask) = mask else { return };
    let m_shape = mask.shape();
    let k_shape = k_full.shape();
    let m_last = m_shape.last().copied().unwrap_or(0);
    let k_last = k_shape[k_shape.len() - 2];
    debug_assert!(
        m_last == k_last,
        "mask key axis ({m_last}) does not match K seq len ({k_last}); \
         mask {m_shape:?}, k_full {k_shape:?}",
    );
}

/// Key-value cache for decoder-only attention.
pub trait KeyValueCache {
    fn is_quantized(&self) -> bool {
        false
    }

    fn group_size(&self) -> Option<i32> {
        None
    }

    fn bits(&self) -> Option<i32> {
        None
    }

    fn offset(&self) -> i32;

    fn max_size(&self) -> Option<i32>;

    fn update_and_fetch(&mut self, keys: Array, values: Array)
        -> Result<(Array, Array), Exception>;

    /// `softmax(scaled_q @ K.T) @ V` over the full cached history.
    fn attention(
        &mut self,
        queries: &Array,
        keys: Array,
        values: Array,
        scale: f32,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        let q_shape = queries.shape();
        let n_q = q_shape[q_shape.len() - 2];
        let (k_full, v_full) = self.update_and_fetch(keys, values)?;
        assert_mask_matches_keys(mask, &k_full);
        scaled_dot_product_attention(
            queries,
            k_full,
            v_full,
            scale,
            resolve_sdpa_mask(mask, n_q),
            None,
        )
    }
}

impl<T> KeyValueCache for &'_ mut T
where
    T: KeyValueCache,
{
    fn is_quantized(&self) -> bool {
        T::is_quantized(self)
    }

    fn group_size(&self) -> Option<i32> {
        T::group_size(self)
    }

    fn bits(&self) -> Option<i32> {
        T::bits(self)
    }

    fn offset(&self) -> i32 {
        T::offset(self)
    }

    fn max_size(&self) -> Option<i32> {
        T::max_size(self)
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        T::update_and_fetch(self, keys, values)
    }

    fn attention(
        &mut self,
        queries: &Array,
        keys: Array,
        values: Array,
        scale: f32,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        T::attention(self, queries, keys, values, scale, mask)
    }
}

/// Pre-allocated, geometrically-grown KV cache.
///
/// Holds `[B, H, capacity, D]` K/V buffers. First call allocates
/// `init_capacity` tokens; overflows double the buffer. Returns
/// graph-view slices over the populated `[..offset]` range, so no
/// per-step `concatenate_axis`.
#[derive(Debug, Clone)]
pub struct KVCache {
    keys: Option<Array>,
    values: Option<Array>,
    offset: i32,
    init_capacity: i32,
}

impl Default for KVCache {
    fn default() -> Self {
        Self::with_init_capacity(DEFAULT_KV_CACHE_INIT_CAPACITY)
    }
}

impl KVCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_init_capacity(init_capacity: i32) -> Self {
        assert!(init_capacity > 0, "KVCache init_capacity must be positive");
        Self {
            keys: None,
            values: None,
            offset: 0,
            init_capacity,
        }
    }

    pub fn init_capacity(&self) -> i32 {
        self.init_capacity
    }

    pub fn capacity(&self) -> i32 {
        match self.keys.as_ref() {
            Some(k) => {
                let shape = k.shape();
                shape[shape.len() - 2]
            }
            None => 0,
        }
    }

    fn alloc_like(template: &Array, capacity: i32) -> Result<Array, Exception> {
        let shape = template.shape();
        let mut buf_shape = shape.to_vec();
        let t_axis = buf_shape.len() - 2;
        buf_shape[t_axis] = capacity;
        zeros_dtype(&buf_shape, template.dtype())
    }

    fn grow_to_fit(&mut self, new_keys: &Array, new_values: &Array) -> Result<(), Exception> {
        let new_shape = new_keys.shape();
        let s = new_shape[new_shape.len() - 2];
        let required = self.offset + s;
        let current_cap = self.capacity();
        if required <= current_cap {
            return Ok(());
        }
        let mut target_cap = if current_cap == 0 {
            self.init_capacity.max(required)
        } else {
            current_cap
        };
        while target_cap < required {
            target_cap *= 2;
        }

        let mut grown_k = Self::alloc_like(new_keys, target_cap)?;
        let mut grown_v = Self::alloc_like(new_values, target_cap)?;

        if let (Some(old_k), Some(old_v)) = (self.keys.take(), self.values.take()) {
            if self.offset > 0 {
                grown_k.try_index_mut(
                    (Ellipsis, 0..self.offset, ..),
                    old_k.index((Ellipsis, 0..self.offset, ..)),
                )?;
                grown_v.try_index_mut(
                    (Ellipsis, 0..self.offset, ..),
                    old_v.index((Ellipsis, 0..self.offset, ..)),
                )?;
            }
        }
        self.keys = Some(grown_k);
        self.values = Some(grown_v);
        Ok(())
    }
}

impl KeyValueCache for KVCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn max_size(&self) -> Option<i32> {
        None
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        let key_shape = keys.shape();
        let s = key_shape[key_shape.len() - 2];

        self.grow_to_fit(&keys, &values)?;

        let buf_k = self.keys.as_mut().expect("allocated by grow_to_fit");
        let buf_v = self.values.as_mut().expect("allocated by grow_to_fit");

        buf_k.try_index_mut((Ellipsis, self.offset..self.offset + s, ..), keys)?;
        buf_v.try_index_mut((Ellipsis, self.offset..self.offset + s, ..), values)?;

        self.offset += s;

        let end = self.offset;
        Ok((
            buf_k.index((Ellipsis, 0..end, ..)),
            buf_v.index((Ellipsis, 0..end, ..)),
        ))
    }
}
