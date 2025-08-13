<?php
// This file is part of Moodle - http://moodle.org/
//
// Moodle is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Moodle is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with Moodle.  If not, see <http://www.gnu.org/licenses/>.

/**
 * Strings for component 'block_terusrag', language 'en'
 *
 * @package    block_terusrag
 * @copyright  2025 Terus e-Learning
 * @author     Khairu Aqsara <khairu@teruselearning.co.uk>
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

$string['pluginname'] = 'Terus RAG';
$string['terusrag:addinstance'] = 'Add a new Terus RAG block';
$string['terusrag:myaddinstance'] = 'Add a new Terus RAG block to the My Moodle page';
$string['terusrag:managesettings'] = 'Manage Terus RAG settings';
$string['blocktitle'] = 'Block title';
$string['blocktitle_help'] = 'The title that appears at the top of the Terus RAG block';

// Settings strings.
$string['geminisettings'] = 'Gemini API Settings';
$string['geminisettings_desc'] = 'Configure the settings for Google Gemini API integration';
$string['gemini_api_key'] = 'Gemini API Key';
$string['gemini_api_key_desc'] = 'Enter your Gemini API key from Google AI Studio';
$string['gemini_endpoint'] = 'Gemini API Endpoint';
$string['gemini_endpoint_desc'] = 'The base URL for Gemini API requests';
$string['gemini_model_chat'] = 'Chat Model';
$string['gemini_model_chat_desc'] = 'Select the Gemini model to use for chat interactions';
$string['gemini_model_embedding'] = 'Embedding Model';
$string['gemini_model_embedding_desc'] = 'Select the model to use for generating embeddings';

$string['vectordbsettings'] = 'Vector Database Settings';
$string['vectordbsettings_desc'] = 'Configure the vector database backend';
$string['vector_database'] = 'Vector Database Type';
$string['vector_database_desc'] = 'Choose which vector database to use for storing embeddings';
$string['vectordb_flatfile'] = 'Moodle DB (Simple)';
$string['vectordb_chromadb'] = 'ChromaDB';
$string['vectordb_supabase'] = 'Supabase';
$string['vectordb_host'] = 'Database Host';
$string['vectordb_host_desc'] = 'The hostname where your vector database is running';
$string['vectordb_port'] = 'Database Port';
$string['vectordb_port_desc'] = 'The port number for connecting to the vector database';
$string['vectordb_username'] = 'Database Username';
$string['vectordb_username_desc'] = 'Username for authenticating with the vector database';
$string['vectordb_password'] = 'Database Password';
$string['vectordb_password_desc'] = 'Password for authenticating with the vector database';
$string['promptsettings'] = 'Prompt Settings';
$string['promptsettings_desc'] = 'Configure system prompts';
$string['system_prompt'] = 'System Prompt';
$string['system_prompt_desc'] = 'Base system prompt for RAG responses (do not remove [the context id] from the prompt)';
$string['system_prompt_default'] = 'You are a Moodle assistant specialized in answering questions about course materials and activities. Your goal is to provide comprehensive and accurate answers using ONLY the provided context.

The context will include Moodle-specific information such as:
- **[Moodle Course ID]**: The main course ID.
- **[Moodle Activity ID]**: The specific ID of a Moodle activity (e.g., a Page, Forum, Assignment).
- **[Type]**: The type of content (e.g., `course_summary`, `page_content`, `forum_intro`, `assign_content`).
- **[Module Type]**: The Moodle module name (e.g., `course`, `page`, `forum`, `assign`).
- **[Source Title]**: The title of the course or activity from which the content originated.

When answering, reference the **most relevant context ID (from the first `[ID]` in the context block)**. Your response MUST strictly follow this format:

**[the_most_relevant_context_id_from_block_terusrag] Your comprehensive answer here, incorporating details about the course, activity, and content type where relevant.**

Separate information related to different topics or sources with a new line for clarity.
If the requested information is not explicitly present in the provided context, or if you are uncertain, you MUST state: "I am sorry, but the information you requested is not available in the provided course materials." Do NOT make up information or use external knowledge.';
$string['stopwords_not_found'] = 'Stop words file not found';
$string['unknowncourse'] = 'Unknown course';
$string['noresultsfound'] = 'No results found';
$string['notokeninformation'] = 'No token information available';

// Scheduled task strings.
$string['datainitializer'] = 'Data Initializer';

// Frontend.
$string['queryplaceholder'] = 'Type your question here...';
$string['responseplaceholder'] = 'Ask a question to get started';
$string['askbutton'] = 'Ask';
$string['token_usage'] = 'Token usage: Prompt: {$a->prompt}, Response: {$a->response}, Total: {$a->total}';

// AI Provider.
$string['aiprovider'] = 'AI Provider';
$string['aiprovider_desc'] = 'Select the AI provider to use for generating responses';

// Prompt Optimize.
$string['optimizeprompt'] = 'Prompt Optimization';
$string['optimizeprompt_desc'] = 'Optimize the system prompt for better AI responses';

// Open AI.
$string['openaisettings'] = 'OpenAI Settings';
$string['openaisettings_desc'] = 'Configure the settings for OpenAI integration';
$string['openai_api_key'] = 'OpenAI API Key';
$string['openai_api_key_desc'] = 'Enter your OpenAI API key';
$string['openai_endpoint'] = 'OpenAI API Endpoint';
$string['openai_endpoint_desc'] = 'The base URL for OpenAI API requests';
$string['openai_model_chat'] = 'Chat Model';
$string['openai_model_chat_desc'] = 'Select the OpenAI model to use for chat interactions';
$string['openai_model_embedding'] = 'Embedding Model';
$string['openai_model_embedding_desc'] = 'Select the model to use for generating embeddings';

// Ollama.
$string['ollamasettings'] = 'Ollama Settings';
$string['ollamasettings_desc'] = 'Configure the settings for Ollama integration';
$string['ollama_api_key'] = 'Ollama API Key';
$string['ollama_api_key_desc'] = 'Enter your Ollama API key';
$string['ollama_endpoint'] = 'Ollama API Endpoint';
$string['ollama_endpoint_desc'] = 'The base URL for Ollama API requests';
$string['ollama_model_chat'] = 'Chat Model';
$string['ollama_model_chat_desc'] = 'Select the Ollama model to use for chat interactions';
$string['ollama_model_embedding'] = 'Embedding Model';
$string['ollama_model_embedding_desc'] = 'Select the model to use for generating embeddings';

