use std::marker::PhantomData;

use mlx_lm_utils::tokenizer::Tokenizer;
use mlx_rs::{error::Exception, module::Module, transforms::eval, Array};

use crate::{
    cache::{KVCache, KeyValueCache},
    error::Error,
    generate::generate_token::{GenerateToken, Stage},
    sampler::{DefaultSampler, Sampler},
    ModelInput, ModelOutput,
};

mod generate_token;

macro_rules! tri {
    ($expr:expr) => {
        match $expr {
            Ok(val) => val,
            Err(e) => return Some(Err(e.into())),
        }
    };
}

pub struct Generate<M, I, S = DefaultSampler, C = KVCache, T = ()> {
    tokenizer: Tokenizer,
    token_generator: GenerateToken<M, I, S, C, T>,
    max_tokens: usize,
    tokens: Vec<Array>,
}

impl Generate<(), ()> {
    pub fn builder() -> Builder<(), (), (), ()> {
        Builder {
            tokenizer: (),
            model: (),
            model_input_marker: PhantomData,
            prompt: (),
            temp: 0.0,
            max_tokens: 256,
            sampler: DefaultSampler,
            cache_marker: PhantomData,
            state: (),
        }
    }
}

pub struct Builder<Tok, M, I, P, S = DefaultSampler, C = KVCache, T = ()> {
    pub tokenizer: Tok,
    pub model: M,
    pub model_input_marker: PhantomData<I>,
    pub prompt: P,
    pub temp: f32,
    pub max_tokens: usize,
    pub sampler: S,
    pub cache_marker: PhantomData<C>,
    pub state: T,
}

impl<Tok, M, I, P, S, C, T> Builder<Tok, M, I, P, S, C, T> {
    pub fn tokenizer(
        self,
        tokenizer: Tokenizer,
    ) -> Builder<Tokenizer, M, I, P, S, C, T> {
        Builder {
            tokenizer,
            model: self.model,
            model_input_marker: self.model_input_marker,
            prompt: self.prompt,
            temp: self.temp,
            max_tokens: self.max_tokens,
            sampler: self.sampler,
            cache_marker: self.cache_marker,
            state: self.state,
        }
    }

    pub fn model<M2, I2>(self, model: M2) -> Builder<Tok, M2, I2, P, S, C, T>
    where
        M2: Module<I2>,
    {
        Builder {
            tokenizer: self.tokenizer,
            model,
            model_input_marker: PhantomData,
            prompt: self.prompt,
            temp: self.temp,
            max_tokens: self.max_tokens,
            sampler: self.sampler,
            cache_marker: self.cache_marker,
            state: self.state,
        }
    }

    pub fn prompt(self, prompt: Array) -> Builder<Tok, M, I, Array, S, C, T> {
        Builder {
            tokenizer: self.tokenizer,
            model: self.model,
            model_input_marker: self.model_input_marker,
            prompt,
            temp: self.temp,
            max_tokens: self.max_tokens,
            sampler: self.sampler,
            cache_marker: self.cache_marker,
            state: self.state,
        }
    }

    pub fn temp(mut self, temp: f32) -> Self {
        self.temp = temp;
        self
    }

    pub fn max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn sampler<S2>(self, sampler: S2) -> Builder<Tok, M, I, P, S2, C, T> {
        Builder {
            tokenizer: self.tokenizer,
            model: self.model,
            model_input_marker: self.model_input_marker,
            prompt: self.prompt,
            temp: self.temp,
            max_tokens: self.max_tokens,
            sampler,
            cache_marker: self.cache_marker,
            state: self.state,
        }
    }

    pub fn cache_marker<C2>(self) -> Builder<Tok, M, I, P, S, C2, T> {
        Builder {
            tokenizer: self.tokenizer,
            model: self.model,
            model_input_marker: self.model_input_marker,
            prompt: self.prompt,
            temp: self.temp,
            max_tokens: self.max_tokens,
            sampler: self.sampler,
            cache_marker: PhantomData,
            state: self.state,
        }
    }

    pub fn state<T2>(self, state: T2) -> Builder<Tok, M, I, P, S, C, T2> {
        Builder {
            tokenizer: self.tokenizer,
            model: self.model,
            model_input_marker: self.model_input_marker,
            prompt: self.prompt,
            temp: self.temp,
            max_tokens: self.max_tokens,
            sampler: self.sampler,
            cache_marker: self.cache_marker,
            state,
        }
    }
}

impl<M, I, S, C, T> Builder<Tokenizer, M, I, Array, S, C, T>
where
    M: Module<I>,
    S: Sampler,
    C: KeyValueCache + Default,
{
    pub fn build(self) -> Generate<M, I, S, C, T> {
        let Self {
            tokenizer,
            model,
            model_input_marker: _,
            prompt,
            temp,
            sampler,
            cache_marker: _,
            state,
            max_tokens,
        } = self;

        let stage = Stage::Prefill { prompt, state };

        let token_generator = GenerateToken {
            model,
            model_input_marker: PhantomData,
            sampler,
            temp,
            stage,
        };

        let tokens = Vec::with_capacity(max_tokens);
        Generate {
            tokenizer,
            token_generator,
            max_tokens,
            tokens,
        }
    }
}

pub struct Response {
    pub text: String,
    pub ids: Vec<u32>,
}

impl<M, I, S, C, T> Iterator for Generate<M, I, S, C, T>
where
    M: Module<I>,
    M::Error: Into<Exception>,
    M::Output: ModelOutput,
    for<'input> I: ModelInput<'input, C, T>,
    S: Sampler,
    C: KeyValueCache + Default,
{
    type Item = Result<Response, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        // Accumulate the lazy (async-submitted) token graphs, then read
        // them back in one host sync — never `.item()` per step, which
        // would stall the pipeline on a coherence barrier every token.
        while self.tokens.len() < self.max_tokens {
            let token = tri!(self.token_generator.next()?);
            self.tokens.push(token);
        }

        tri!(eval(self.tokens.iter()));
        let ids: Vec<u32> = self
            .tokens
            .drain(..)
            .map(|t| t.item::<u32>())
            .collect();
        let text = tri!(self.tokenizer.decode(&ids, true));
        Some(Ok(Response { text, ids }))
    }
}
