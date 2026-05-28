use std::marker::PhantomData;

use mlx_rs::{error::Exception, module::Module, transforms::async_eval, Array};

use crate::{
    cache::KeyValueCache, sampler::Sampler, ModelInput, ModelInputBuilder, ModelOutput,
};

macro_rules! tri {
    ($expr:expr) => {
        match $expr {
            Ok(val) => val,
            Err(e) => return Some(Err(e.into())),
        }
    };
}

pub(super) enum Stage<C, T> {
    Generating,
    Prefill {
        prompt: Array,
        state: T,
    },
    /// `last` not yet forwarded; first decode call primes the pipeline.
    DecodeFirst {
        last: Array,
        cache: Vec<Option<C>>,
        state: T,
    },
    /// `pending` already async-submitted; each step submits N+1 before
    /// yielding N so the consumer's sync of N overlaps N+1.
    Decode {
        pending: Array,
        cache: Vec<Option<C>>,
        state: T,
    },
}

impl<C, T> Stage<C, T> {
    fn take(&mut self) -> Self {
        debug_assert!(!matches!(self, Self::Generating));

        let mut swap = Self::Generating;
        std::mem::swap(self, &mut swap);
        swap
    }
}

pub(super) struct GenerateToken<M, I, S, C, T> {
    pub model: M,
    pub model_input_marker: PhantomData<I>,
    pub sampler: S,
    pub temp: f32,
    pub stage: Stage<C, T>,
}

impl<M, I, S, C, T> GenerateToken<M, I, S, C, T>
where
    M: Module<I>,
    M::Error: Into<Exception>,
    M::Output: ModelOutput,
    for<'input> I: ModelInput<'input, C, T>,
    S: Sampler,
    C: KeyValueCache + Default,
{
    /// Forward + sample on `y`, async-submitted; returns the next token.
    fn step(
        &mut self,
        y: &Array,
        cache: &mut Vec<Option<C>>,
        state: &mut T,
    ) -> Result<Array, Exception> {
        let builder = ModelInputBuilder { y, cache, state };
        let input = I::from_model_input_builder(builder);
        let output = self.model.forward(input).map_err(Into::into)?;
        let next = self.sampler.sample(output.logits(), self.temp)?;
        async_eval([&next])?;
        Ok(next)
    }
}

impl<M, I, S, C, T> Iterator for GenerateToken<M, I, S, C, T>
where
    M: Module<I>,
    M::Error: Into<Exception>,
    M::Output: ModelOutput,
    for<'input> I: ModelInput<'input, C, T>,
    S: Sampler,
    C: KeyValueCache + Default,
{
    type Item = Result<Array, Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.stage.take() {
            Stage::Prefill { prompt, mut state } => {
                let mut cache = Vec::new();
                let first = tri!(self.step(&prompt, &mut cache, &mut state));
                self.stage = Stage::DecodeFirst {
                    last: first.clone(),
                    cache,
                    state,
                };
                Some(Ok(first))
            }
            Stage::DecodeFirst {
                last,
                mut cache,
                mut state,
            } => {
                let next = tri!(self.step(&last, &mut cache, &mut state));
                let pending = tri!(self.step(&next, &mut cache, &mut state));
                self.stage = Stage::Decode {
                    pending,
                    cache,
                    state,
                };
                Some(Ok(next))
            }
            Stage::Decode {
                pending,
                mut cache,
                mut state,
            } => {
                let next = tri!(self.step(&pending, &mut cache, &mut state));
                self.stage = Stage::Decode {
                    pending: next,
                    cache,
                    state,
                };
                Some(Ok(pending))
            }
            Stage::Generating => unreachable!(),
        }
    }
}
