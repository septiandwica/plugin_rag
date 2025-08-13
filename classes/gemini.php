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
 * Gemini provider implementation.
 *
 * @package    block_terusrag
 * @copyright  2025 Terus e-Learning
 * @author     khairu@teruselearning.co.uk
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

namespace block_terusrag;

use curl;
use moodle_exception;
use stdClass; // Import stdClass
use context_module;
use course_modinfo;
use core_completion\course_module_completion; // For completion status if needed
use html_writer; // For cleaning HTML content

/**
 * Gemini API provider implementation for the TerusRAG block.
 */
class gemini implements provider_interface {

    /** @var string API key for Gemini services */
    protected string $apikey;

    /** @var string Base URL for the Gemini API */
    protected string $host;

    /** @var string Model name for chat functionality */
    protected string $chatmodel;

    /** @var string Model name for embedding functionality */
    protected string $embeddingmodel;

    /** @var array HTTP headers for API requests */
    protected array $headers;

    /** @var curl HTTP client for API communication */
    protected curl $httpclient;

    /** @var string System prompt to guide model behavior */
    protected string $systemprompt;

    /** @var bool Whether to prompt for optimization */
    protected bool $promptoptimization = false;

    /**
     * Constructor initializes the Gemini API client.
     */
    public function __construct() {
        $apikey = get_config("block_terusrag", "gemini_api_key");
        $host = get_config("block_terusrag", "gemini_endpoint");
        $embeddingmodels = get_config(
            "block_terusrag",
            "gemini_model_embedding"
        );
        $chatmodels = get_config("block_terusrag", "gemini_model_chat");
        $systemprompt = get_config("block_terusrag", "system_prompt");
        $promptoptimization = get_config("block_terusrag", "optimizeprompt");

        $this->systemprompt = $systemprompt;
        $this->apikey = $apikey;
        $this->host = $host;
        $this->chatmodel = $chatmodels;
        $this->embeddingmodel = $embeddingmodels;
        $this->promptoptimization = $promptoptimization === 'yes' ? true : false;
        $this->headers = [
            "Content-Type: application/json",
            "x-goog-api-key: " . $this->apikey,
        ];
        $this->httpclient = new curl([
            "cache" => true,
            "module_cache" => "terusrag",
        ]);
        $this->httpclient->setHeader($this->headers);
        $this->httpclient->setopt([
            "CURLOPT_SSL_VERIFYPEER" => false,
            "CURLOPT_SSL_VERIFYHOST" => false,
            "CURLOPT_TIMEOUT" => 30,
            "CURLOPT_CONNECTTIMEOUT" => 30,
        ]);
    }

    /**
     * Generate embedding vectors for the given text query.
     *
     * @param string|array $query Text to generate embeddings for
     * @return array Array of embedding values
     * @throws moodle_exception If API request fails
     */
    public function get_embedding($query) {
        $query = is_array($query) ? $query : [$query];
        $payload = [
            "requests" => array_map(function ($text) {
                return [
                    "model" => "models/" . $this->embeddingmodel,
                    "content" => [
                        "parts" => [
                            [
                                "text" => $text,
                            ],
                        ],
                    ],
                ];
            }, $query),
        ];

        $response = $this->httpclient->post(
            $this->host .
                "/v1beta/models/" .
                $this->embeddingmodel .
                ":batchEmbedContents",
            json_encode($payload)
        );

        if ($this->httpclient->get_errno()) {
            $error = $this->httpclient->error;
            debugging("Curl error: " . $error);
            throw new moodle_exception("Curl error: " . $error);
        }

        $data = json_decode($response, true);

        if (json_last_error() !== JSON_ERROR_NONE) {
            throw new moodle_exception(
                "JSON decode error: " . json_last_error_msg()
            );
        }

        if (isset($data["embeddings"]) && is_array($data["embeddings"])) {
            $embeddingsdata = $data["embeddings"];
            return is_array($query)
                ? $embeddingsdata
                : $embeddingsdata[0]["values"];
        } else {
            debugging("Gemini API: Invalid response format: " . $response);
            throw new moodle_exception("Invalid response from Gemini API");
        }
    }

    /**
     * Get a response from the Gemini chat model.
     *
     * @param string $prompt The prompt to send to the model
     * @return array The response data from the API
     * @throws moodle_exception If the API request fails
     */
    public function get_response($prompt) {
        $payload = [
            "contents" => [
                "parts" => [
                    [
                        "text" => $prompt,
                    ],
                ],
            ],
        ];

        $response = $this->httpclient->post(
            $this->host .
                "/v1beta/models/" .
                $this->chatmodel .
                ":generateContent?key=" .
                $this->apikey,
            json_encode($payload)
        );

        if ($this->httpclient->get_errno()) {
            $error = $this->httpclient->error;
            debugging("Curl error: " . $error);
            throw new moodle_exception("Curl error: " . $error);
        }

        $data = json_decode($response, true);

        if (json_last_error() !== JSON_ERROR_NONE) {
            throw new moodle_exception(
                "JSON decode error: " . json_last_error_msg()
            );
        }

        return $data;
    }

    /**
     * Process a RAG query with the Gemini model.
     *
     * @param string $userquery The user's query
     * @return array The processed response
     */
    public function process_rag_query(string $userquery) {
        global $DB;

        // 1. System Prompt (Define role and behavior).
        $systemprompt = $this->systemprompt;

        // 1.1 Optimize User Prompt.
        if ($this->promptoptimization) {
            $llm = new llm();
            $optimizedquery = $llm->optimize_prompt($userquery);
            $userquery = (isset($optimizedquery["optimized_prompt"]) && !empty($optimizedquery["optimized_prompt"]))
                ? $optimizedquery["optimized_prompt"]
                : $userquery;
        }

        // 2. Retrieve relevant chunks.
        $toprankchunks = $this->get_top_ranked_chunks($userquery);

        // 3. Context Injection.
        // Enhance context injection with more Moodle-aware information
        $contextinjection = "Context:\n";
        foreach ($toprankchunks as $chunk) {
            $contextinjection .= "[Moodle Course ID: {$chunk['moduleid']}] ";
            if (!empty($chunk['activityid'])) {
                $contextinjection .= "[Moodle Activity ID: {$chunk['activityid']}] ";
            }
            $contextinjection .= "[Type: {$chunk['contenttype']}] "; // e.g., course_summary, page_content, forum_intro
            $contextinjection .= "[Module Type: {$chunk['moduletype']}] "; // e.g., course, page, forum
            $contextinjection .= "[Source Title: {$chunk['title']}]\n";
            $contextinjection .= $chunk['content'] . "\n\n";
        }


        // 4. User Query.
        $prompt =
            $systemprompt .
            "\n" .
            $contextinjection .
            "Question: " .
            $userquery .
            "\nAnswer:";

        // 5. API Call to Gemini.
        $answer = $this->get_response($prompt);
        $response = [
            "answer" => isset($answer["candidates"])
                ? $this->parse_response($answer["candidates"])
                : [],
            "promptTokenCount" => isset(
                $answer["usageMetadata"]["promptTokenCount"]
            )
                ? $answer["usageMetadata"]["promptTokenCount"]
                : 0,
            "responseTokenCount" => isset(
                $answer["usageMetadata"]["candidatesTokenCount"]
            )
                ? $answer["usageMetadata"]["candidatesTokenCount"]
                : 0,
            "totalTokenCount" => isset(
                $answer["usageMetadata"]["totalTokenCount"]
            )
                ? $answer["usageMetadata"]["totalTokenCount"]
                : 0,
        ];

        // Log if unexpected response structure is received.
        if (!isset($answer["candidates"]) || !isset($answer["usageMetadata"])) {
            debugging(
                "Gemini API returned unexpected response structure: " .
                    json_encode($answer),
                DEBUG_DEVELOPER
            );
        }

        return $response;
    }

    /**
     * Extract all text from a nested response array.
     *
     * @param array $array The response array to process
     * @param string $result The accumulated result string
     * @return string The extracted text
     */
    public function extract_all_text_response($array, &$result = "") {
        foreach ($array as $key => $value) {
            if ($key === "text" && is_string($value)) {
                $result .= $value . " ";
            } else if (is_array($value)) {
                $this->extract_all_text_response($value, $result);
            }
        }
        return $result;
    }

    /**
     * Parse the response from the Gemini API.
     *
     * @param array $response The response from the API
     * @return array Parsed response as an array of lines
     */
    public function parse_response($response) {
        $text = $this->extract_all_text_response($response);
        $text = trim($text);
        // Split by newline and clean up each line.
        $lines = explode("\n", $text);
        $cleanlines = [];

        foreach ($lines as $line) {
            $line = trim($line);
            if (!empty($line)) {
                $cleanlines[] = $this->get_course_from_proper_answer(
                    $this->get_proper_answer($line)
                );
            }
        }

        // Filter out items where id is 0 or not set.
        return array_filter($cleanlines, function ($item) {
            return isset($item["id"]) && $item["id"] != 0;
        });
    }

    /**
     * Get course information from a properly formatted answer.
     *
     * This function now aims to find the relevant Moodle item (course or activity)
     * based on the ID provided in the LLM's answer.
     *
     * @param array $response The formatted response array containing 'id' and 'content'
     * @return array Moodle item information with id, title, content, and view URL
     */
    public function get_course_from_proper_answer(array $response) {
        global $DB;
        $id = $response["id"] ?? 0;
        $content = $response["content"] ?? '';

        if ($id === 0) {
            return [
                "id" => 0,
                "title" => "Unknown Source",
                "content" => $content,
                "viewurl" => null,
            ];
        }

        // First, try to find a record in block_terusrag to get the contenttype and moduletype.
        $terusragrecord = $DB->get_record('block_terusrag', ['id' => $id]);

        if ($terusragrecord) {
            $moduletype = $terusragrecord->moduletype;
            $moduleid = $terusragrecord->moduleid; // This is the Course ID for course_summary, or activity's course ID
            $activityid = $terusragrecord->activityid; // This is the cmid if available

            $title = $terusragrecord->title;
            $viewurl = null;

            if ($moduletype === 'course') {
                $course = $DB->get_record("course", ["id" => $moduleid]);
                if ($course) {
                    $title = $course->fullname;
                    $viewurl = new \moodle_url("/course/view.php", ["id" => $moduleid]);
                }
            } else if (!empty($activityid)) {
                // It's an activity, try to get the URL for the specific activity.
                $cm = get_course_and_cm_from_cmid($activityid, $moduleid); // $moduleid is courseid here
                if ($cm && $cm->cm) {
                    $url = new \moodle_url("/mod/{$cm->modname}/view.php", ['id' => $cm->cm->id]);
                    $viewurl = $url;
                    $title = $cm->cm->name; // Use the activity name
                } else {
                    // Fallback to course view if activity not found or no specific URL
                    $course = $DB->get_record("course", ["id" => $moduleid]);
                    if ($course) {
                        $title = $course->fullname . " (Activity in course)";
                        $viewurl = new \moodle_url("/course/view.php", ["id" => $moduleid]);
                    }
                }
            } else {
                // It's from a module type but no specific activity ID, likely still linked to a course.
                $course = $DB->get_record("course", ["id" => $moduleid]);
                if ($course) {
                    $title = $course->fullname;
                    $viewurl = new \moodle_url("/course/view.php", ["id" => $moduleid]);
                }
            }

            return [
                "id" => $id, // This ID is the block_terusrag ID, not Moodle's
                "title" => $title,
                "content" => $content,
                "viewurl" => !is_null($viewurl) ? $viewurl->out() : null,
            ];
        }

        // Fallback if no record found in block_terusrag (e.g., if LLM hallucinates an ID)
        return [
            "id" => 0,
            "title" => "Unknown Source",
            "content" => $content,
            "viewurl" => null,
        ];
    }


    /**
     * Format a string answer into a structured response.
     * This function now extracts the ID from the `block_terusrag` table.
     *
     * @param string $originalstring The original response string from the LLM
     * @return array Structured response with ID (from block_terusrag) and content
     */
    public function get_proper_answer($originalstring) {
        // Updated regex to find an ID that is likely to be a block_terusrag ID
        // The LLM should ideally provide the block_terusrag ID it used in context.
        // Example: "The answer is [ID] This is the content..."
        preg_match("/\[(\d+)\]/", $originalstring, $matches); // Look for [ID]
        $id = isset($matches[1]) ? (int) $matches[1] : null;

        // Clean the string by removing the [ID] part.
        $cleanstring = preg_replace("/^\[\d+\]\s*/", "", $originalstring);
        $cleanstring = preg_replace("/\s*\[\d+\]\s*$/", "", $cleanstring); // Also remove if at end

        return ["id" => $id, "content" => $cleanstring];
    }

    /**
     * Get the top ranked content chunks for a given query.
     *
     * @param string $query The search query
     * @return array The top-ranked content chunks
     */
    public function get_top_ranked_chunks(string $query): array {
        global $DB;

        // Generate embedding for the query.
        $queryembeddingresponse = $this->get_embedding($query);
        $queryembedding = $queryembeddingresponse[0]["values"];

        // Process chunks in batches to manage memory.
        $batchsize = 500;
        $chunkscores = [];
        $bm25scores = [];
        $contentarray = []; // This will store full chunk data, not just content
        // Adjusted SQL query to select all necessary fields, matching install.xml
        $sql = "SELECT id, content, embedding, moduleid, moduletype, contenttype, title, activityid FROM {block_terusrag}";
        $rs = $DB->get_recordset_sql($sql);

        try {
            $batch = [];
            $llm = new llm(); // Ensure this class exists or is properly autoloaded
            $documents = [];

            // First pass - collect documents for BM25 indexing.
            foreach ($rs as $record) {
                $documents[$record->id] = $record->content;
                // Store the entire record for later retrieval if it's a top chunk
                $contentarray[$record->id] = (array)$record;
            }

            // Initialize BM25 with collected documents.
            $bm25 = new bm25($documents); // Ensure this class exists

            // Reset recordset for second pass.
            $rs->close();
            $rs = $DB->get_recordset_sql($sql);

            // Second pass - process chunks.
            foreach ($rs as $record) {
                $batch[] = $record;
                // $contentarray[$record->id] = (array)$record; // Already done in first pass
                if (count($batch) >= $batchsize) {
                    $this->process_chunk_batch($batch, $queryembedding, $query, $llm, $bm25, $chunkscores, $bm25scores);
                    $batch = [];
                    gc_collect_cycles();
                }
            }

            // Process remaining chunks.
            if (!empty($batch)) {
                $this->process_chunk_batch($batch, $queryembedding, $query, $llm, $bm25, $chunkscores, $bm25scores);
            }

            // Hybrid scoring and ranking.
            $hybridscores = [];
            foreach ($chunkscores as $chunkid => $cosinesimilarity) {
                $bm25score = $bm25scores[$chunkid] ?? 0;
                // Adjust weights if necessary. Example: 0.7 for cosine, 0.3 for BM25.
                $hybridscores[$chunkid] = 0.7 * $cosinesimilarity + 0.3 * $bm25score;
            }
            arsort($hybridscores); // Sort in descending order of score

            // Select top 5 chunks.
            $topnchunkids = array_slice(array_keys($hybridscores), 0, 5, true);
            $topnchunks = [];

            foreach ($topnchunkids as $chunkid) {
                // Return the full chunk data, including Moodle IDs and types
                $topnchunks[] = $contentarray[$chunkid];
            }

            return $topnchunks;

        } finally {
            $rs->close();
        }
    }

    /**
     * Process a batch of chunks for similarity scoring and ranking.
     *
     * @param array $batch Array of database records containing chunks to process
     * @param array $queryembedding Query embedding vector for similarity comparison as a numeric array
     * @param string $query Original query string for BM25 scoring
     * @param \block_terusrag\llm $llm LLM helper instance for similarity calculations
     * @param \block_terusrag\bm25 $bm25 BM25 ranking instance for text relevancy scoring
     * @param array &$chunkscores Reference to array storing chunk similarity scores, indexed by chunk ID
     * @param array &$bm25scores Reference to array storing BM25 scores, indexed by chunk ID
     * @return void
     */
    protected function process_chunk_batch($batch, $queryembedding, $query, $llm, $bm25, &$chunkscores, &$bm25scores) {
        foreach ($batch as $chunk) {
            $chunkembedding = unserialize($chunk->embedding);
            if ($chunkembedding && is_array($chunkembedding)) { // Ensure it's a valid array
                $chunkscores[$chunk->id] = $llm->cosine_similarity($queryembedding, $chunkembedding);
                $bm25scores[$chunk->id] = $bm25->score($query, $chunk->content, $chunk->id);
            } else {
                // Log or handle invalid embedding data
                debugging("Invalid embedding data for chunk ID: " . $chunk->id, DEBUG_DEVELOPER);
                $chunkscores[$chunk->id] = 0;
                $bm25scores[$chunk->id] = 0;
            }
        }
    }

    /**
     * Initializes data by processing courses, chunking content, and generating embeddings.
     * This method retrieves visible courses, processes their content into chunks,
     * generates embeddings for each chunk, and stores the data in the database.
     *
     * @return void
     */
    public function data_initialization() {
        global $DB, $CFG;

        $batchsize = 100;
        $lastprocessedtime = get_config('block_terusrag', 'last_processed_time') ?? 0;
        $currenttime = time();
        $chunksize = 512; // Ideal chunk size for embeddings

        // Retrieve courses that have been modified since the last processing.
        $sql = "SELECT c.id, c.fullname, c.summary, c.timemodified FROM {course} c WHERE c.visible = 1 AND c.timemodified > ?";
        $rs = $DB->get_recordset_sql($sql, [$lastprocessedtime]);

        $coursebatch = [];

        try {
            foreach ($rs as $course) {
                // Add course summary as a chunk.
                $coursecontent = trim(strip_tags($course->summary ?: $course->fullname));
                if (!empty($coursecontent)) {
                    $coursebatch[] = [
                        'content' => $coursecontent,
                        'title' => $course->fullname,
                        'moduleid' => $course->id,
                        'timemodified' => $course->timemodified,
                        'contenttype' => 'course_summary',
                        'moduletype' => 'course',
                        'activityid' => null,
                    ];
                }

                // --- Process Course Modules (Activities and Resources) ---
                $modinfo = get_fast_modinfo($course->id);

                foreach ($modinfo->get_cms() as $cm) {
                    // Skip unavailable or hidden modules if you only want visible content
                    if (!$cm->uservisible || !$cm->available) {
                        continue;
                    }

                    $content = '';
                    $title = $cm->name;
                    $moduletype = $cm->modname;
                    $contenttype = $moduletype . '_content';

                    switch ($moduletype) {
                        case 'page':
                            $page = $DB->get_record('page', ['id' => $cm->instance]);
                            if ($page) {
                                $content = trim(strip_tags($page->content . ' ' . $page->summary));
                            }
                            break;
                        case 'forum':
                            $forum = $DB->get_record('forum', ['id' => $cm->instance]);
                            if ($forum) {
                                $content = trim(strip_tags($forum->intro)); // Forum introduction
                                // OPTIONAL: Index individual forum discussions/posts
                                // This can generate a lot of data. You might want to do this in a separate, more granular process.
                                // $this->process_forum_discussions($forum->id, $cm->course);
                            }
                            break;
                        case 'assign':
                            $assign = $DB->get_record('assign', ['id' => $cm->instance]);
                            if ($assign) {
                                $content = trim(strip_tags($assign->intro)); // Assignment description
                            }
                            break;
                        case 'quiz':
                            $quiz = $DB->get_record('quiz', ['id' => $cm->instance]);
                            if ($quiz) {
                                $content = trim(strip_tags($quiz->intro)); // Quiz introduction
                                // You might also consider indexing quiz questions if accessible and relevant.
                            }
                            break;
                        case 'resource':
                            $resource = $DB->get_record('resource', ['id' => $cm->instance]);
                            if ($resource) {
                                $content = trim(strip_tags($resource->intro)); // Resource description
                                // For file resources, you might need to extract text from PDFs/DOCs (complex!)
                            }
                            break;
                        case 'url':
                            $url = $DB->get_record('url', ['id' => $cm->instance]);
                            if ($url) {
                                $content = trim(strip_tags($url->intro)); // URL description
                            }
                            break;
                        case 'book':
                            $book = $DB->get_record('book', ['id' => $cm->instance]);
                            if ($book) {
                                $content = trim(strip_tags($book->intro)); // Book intro
                                // OPTIONAL: Index book chapters (requires fetching from book_chapters table)
                                // $this->process_book_chapters($book->id, $cm->course);
                            }
                            break;
                        // Add more module types as needed: wiki, glossary, lesson, folder, etc.
                        default:
                            // For unknown or unsupported module types, you might still want to index the CM's name/intro
                            $content = trim(strip_tags($cm->intro));
                            if (empty($content)) {
                                $content = $title; // Fallback to title if intro is empty
                            }
                            $contenttype = $moduletype . '_content'; // Use module type as content type
                            break;
                    }

                    if (!empty($content)) {
                        $coursebatch[] = [
                            'content' => $content,
                            'title' => $title,
                            'moduleid' => $course->id, // This is the Course ID
                            'timemodified' => $cm->timemodified,
                            'contenttype' => $contenttype,
                            'moduletype' => $moduletype,
                            'activityid' => $cm->id, // This is the course module ID (cmid)
                        ];
                    }

                    // Process batches to avoid out-of-memory errors
                    if (count($coursebatch) >= $batchsize) {
                        $this->process_course_batch($coursebatch, $chunksize);
                        $coursebatch = [];
                        gc_collect_cycles(); // Explicitly free memory
                    }
                }
            }

            // Process any remaining courses/modules.
            if (!empty($coursebatch)) {
                $this->process_course_batch($coursebatch, $chunksize);
            }

            // Update last processed time ONLY if we successfully processed some data.
            set_config('last_processed_time', $currenttime, 'block_terusrag');

        } finally {
            $rs->close(); // Ensure recordset is always closed.
        }
    }


    /**
     * Process a batch of courses and their modules for embedding generation.
     * This function now also handles activity-specific IDs and content types.
     *
     * @param array $coursebatch Array of course/module data to process
     * @param int $chunksize Size of content chunks
     * @return void
     */
    protected function process_course_batch($coursebatch, $chunksize) {
        global $DB;

        $chunks_for_embedding = [];
        $chunk_metadata_map = [];  // Maps chunk index to course/activity data.

        // Prepare chunks for batch embedding.
        foreach ($coursebatch as $dataentry) {
            $string = $dataentry['content'];
            $stringlength = mb_strlen($string);

            // Chunk the content based on chunksize.
            for ($i = 0; $i < $stringlength; $i += $chunksize) {
                $chunk = mb_substr($string, $i, $chunksize);
                $chunkindex = count($chunks_for_embedding);
                $chunks_for_embedding[] = $chunk;

                // Store full metadata for each chunk/sub-chunk
                $chunk_metadata_map[$chunkindex] = [
                    'title' => $dataentry['title'],
                    'moduleid' => $dataentry['moduleid'], // Course ID
                    'content' => $chunk,
                    'contenthash' => sha1($chunk), // Recalculate hash for the chunk
                    'timemodified' => $dataentry['timemodified'],
                    'contenttype' => $dataentry['contenttype'],
                    'moduletype' => $dataentry['moduletype'],
                    'activityid' => $dataentry['activityid'], // CMID if applicable
                ];
            }
        }

        // Get embeddings for all chunks in batch.
        if (!empty($chunks_for_embedding)) {
            $embeddingsdata = $this->get_embedding($chunks_for_embedding);

            // Process each embedding and update database.
            foreach ($embeddingsdata as $i => $embedding_result) {
                if (!isset($chunk_metadata_map[$i])) {
                    continue; // Should not happen
                }

                $chunkdata = $chunk_metadata_map[$i];
                // Ensure embedding_result is an array and has 'values' key, or is the direct array.
                $values = is_array($embedding_result) && isset($embedding_result["values"]) ? $embedding_result["values"] : $embedding_result;

                // If $values is not an array (e.g., API returned error or empty array), skip this embedding.
                if (!is_array($values) || empty($values)) {
                    debugging("Empty or invalid embedding values for chunk index: {$i}. Skipping.", DEBUG_DEVELOPER);
                    continue;
                }

                $chunkdata['embedding'] = serialize($values);
                $chunkdata['timecreated'] = time();
                $chunkdata['timemodified'] = time();

                // Convert to stdClass for Moodle DB functions
                $record_to_save = new stdClass();
                foreach ($chunkdata as $key => $value) {
                    // Ensure null values are handled correctly for database (don't set if null for NOT NULL fields)
                    if ($key === 'activityid' && is_null($value)) {
                        $record_to_save->$key = null; // Store NULL for nullable fields
                    } else {
                        $record_to_save->$key = $value;
                    }
                }

                // Check if record exists based on contenthash, moduleid, and contenttype for uniqueness.
                $existingrecord = $DB->get_record('block_terusrag', [
                    'contenthash' => $record_to_save->contenthash,
                    'moduleid' => $record_to_save->moduleid,
                    'contenttype' => $record_to_save->contenttype,
                ]);

                if ($existingrecord) {
                    $record_to_save->id = $existingrecord->id;
                    $DB->update_record('block_terusrag', $record_to_save);
                } else {
                    $DB->insert_record('block_terusrag', $record_to_save);
                }
            }
        }
    }
}