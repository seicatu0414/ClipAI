# ドメイン

特定のAI実装ではなく、長く変わらない業務上の意味を表す名前を使います。

- **Streamer**：学習対象となる配信者。内部ID、YouTubeチャンネルID・URL、表示名を持ち、全履歴を巨大な単一オブジェクトとして保持しません。
- **Stream**：1本の動画または配信アーカイブ。1つのStreamを複数回分析できます。
- **AnalysisSession**：1つのStreamに対する再現可能な分析実行。パイプライン、モデル、プロンプト、設定、状態、成果物、エラー、時間を記録し、比較に必要な履歴を保持します。
- **Transcript**：開始・終了時刻、本文、任意の話者、信頼度情報を持つ順序付き発話区間です。
- **Event**：笑い、叫び、歌、感情的な声、異常な無音、勝敗、印象的発言、視聴者反応、コールバックなどの証拠であり、自動的に候補になるものではありません。
- **StreamerKnowledge**：配信者の発話傾向、感情基準、定番表現・ネタ、行動、強み、コラボ、好み、コールバックについて、信頼度と証拠を伴うバージョン付き知識です。過去を削除せず、断定を避けます。
- **ClipCandidate**：開始・終了、カテゴリ別・総合スコア、信頼度、理由、証拠、分析元、レビュー状態を持つ提案です。
- **Feedback**：候補に対する評価、理由タグ、任意メモ、時刻です。過去の候補を書き換えず将来の好みに影響します。

StreamはStreamerに、AnalysisSessionはStreamに属します。Transcript、Event、ClipCandidateはAnalysisSessionに、FeedbackはClipCandidateに属します。StreamerKnowledgeはStreamerに属し、AnalysisSession由来の証拠を参照します。
