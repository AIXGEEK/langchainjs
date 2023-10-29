import { ChatGLM } from "langchain/chat_models/chatglm";
import { HumanMessage } from "langchain/schema";

const model = new ChatGLM();

const messages1 = [new HumanMessage("Nice to meet you!")];

const res1 = await model.call(messages1);

console.log(res1);

const messages2 = [new HumanMessage("Hello")];

const res2 = await model.call(messages2);

console.log(res2);
