import { SignJWT } from "jose";
import { CallbackManagerForLLMRun } from "../callbacks/manager.js";
import {
  AIMessage,
  BaseMessage,
  ChatGeneration,
  ChatMessage,
  ChatResult,
} from "../schema/index.js";
import { getEnvironmentVariable } from "../util/env.js";
import { IterableReadableStream } from "../util/stream.js";
import { BaseChatModel, BaseChatModelParams } from "./base.js";

type ChatGLMMessageRole = "assistant" | "user";

interface ChatGLMMessage {
  role: ChatGLMMessageRole;
  content: string;
}

interface ChatCompletionRequest {
  prompt: ChatGLMMessage[];
}

interface ChatCompletionResponse {
  code: number;
  success: boolean;
  msg: string;
  data: {
    request_id: string;
    task_id: string;
    task_status: string;
    choices: ChatGLMMessage[];
    usage?: {
      completion_tokens: number;
      prompt_tokens: number;
      total_tokens: number;
    };
  };
}

function extractGenericMessageCustomRole(message: ChatMessage) {
  if (message.role !== "assistant" && message.role !== "user") {
    console.warn(`Unknown message role: ${message.role}`);
  }
  return message.role as ChatGLMMessageRole;
}

function messageToChatGLMRole(message: BaseMessage): ChatGLMMessageRole {
  const type = message._getType();
  switch (type) {
    case "ai":
      return "assistant";
    case "human":
      return "user";
    case "system":
      throw new Error("System messages should not be here");
    case "function":
      throw new Error("Function messages not supported");
    case "generic": {
      if (!ChatMessage.isInstance(message))
        throw new Error("Invalid generic chat message");
      return extractGenericMessageCustomRole(message);
    }
    default:
      throw new Error(`Unknown message type: ${type}`);
  }
}

export interface ChatGLMInput {
  chatGLMApiKey?: string;
  expSeconds?: number;
}

export class ChatGLM extends BaseChatModel implements ChatGLMInput {
  get callKeys(): string[] {
    return ["stop", "signal", "options"];
  }

  get lc_aliases(): { [key: string]: string } | undefined {
    return undefined;
  }

  lc_serializable = true;

  chatGLMApiKey: string;

  expSeconds = 3600;

  static lc_name() {
    return "ChatGLM";
  }

  constructor(fields?: Partial<ChatGLMInput> & BaseChatModelParams) {
    super(fields ?? {});

    const chatGLMApiKey =
      fields?.chatGLMApiKey ?? getEnvironmentVariable("CHATGLM_API_KEY");
    if (!chatGLMApiKey) {
      throw new Error("ChatGLM API key not found");
    } else {
      this.chatGLMApiKey = chatGLMApiKey;
    }

    this.expSeconds = fields?.expSeconds ?? this.expSeconds;
  }

  /** @ignore */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  _combineLLMOutput(): Record<string, any> | undefined {
    return [];
  }

  _llmType(): string {
    return "chatglm";
  }

  async _getJWTToken() {
    const [id, secret] = this.chatGLMApiKey.split(".");
    const payload = {
      api_key: id,
      exp: Math.round(Date.now() / 1000) + this.expSeconds * 1000,
      timestamp: Math.round(Date.now() / 1000),
    };
    const header = {
      alg: "HS256",
      sign_type: "SIGN",
    };
    const jwt = new SignJWT(payload)
      .setProtectedHeader(header)
      .setExpirationTime(payload.exp);
    const token = await jwt.sign(new TextEncoder().encode(secret));
    return token;
  }

  async completion(
    request: ChatCompletionRequest,
    stream: true,
    signal?: AbortSignal
  ): Promise<IterableReadableStream<Uint8Array>>;

  async completion(
    request: ChatCompletionRequest,
    stream: false,
    signal?: AbortSignal
  ): Promise<ChatCompletionResponse>;

  async completion(
    request: ChatCompletionRequest,
    stream: boolean,
    signal?: AbortSignal
  ) {
    const url = `https://open.bigmodel.cn/api/paas/v3/model-api/chatglm_turbo/invoke`;
    const response = await fetch(url, {
      method: "POST",
      headers: {
        Authorization: await this._getJWTToken(),
        accept: stream ? "text/event-stream" : "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
      signal,
    });
    if (stream) {
      if (response.body) {
        const streams = IterableReadableStream.fromReadableStream(
          response.body
        );
        return streams;
      }
    } else {
      return await response.json();
    }
  }

  async _generate(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    _runManager?: CallbackManagerForLLMRun | undefined
  ): Promise<ChatResult> {
    const messagesMapped: ChatGLMMessage[] = messages.map((message) => ({
      role: messageToChatGLMRole(message),
      content: message.content,
    }));
    const result = await this.completion(
      { prompt: messagesMapped },
      false,
      options.signal
    );
    const generations: ChatGeneration[] = [];
    const text = result.data?.choices[0]?.content;
    generations.push({
      text: text,
      message: new AIMessage(text),
    });
    return {
      generations,
      llmOutput: { tokenUsage: result.data?.usage },
    };
  }
}
