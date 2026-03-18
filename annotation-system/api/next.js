import { supabase } from "./lib/supabase.js";

export default async function handler(req, res) {
  const { token } = req.query;

  if (!token) return res.status(400).json({ error: "token required" });

  // validar token
  const { data: annotator } = await supabase
    .from("annotators")
    .select("*")
    .eq("token", token)
    .single();

  if (!annotator) return res.status(403).json({ error: "invalid token" });

  // buscar próximo assignment
  const { data: assignment } = await supabase
    .from("assignments")
    .select(
      `
        id,
        dataset_comments (
            id,
            video_title,
            comment
        )
    `,
    )
    .eq("token", token)
    .eq("completed", false)
    .limit(1)
    .single();

  if (!assignment) {
    return res.status(200).json({ done: true });
  }

  res.status(200).json({
    assignment_id: assignment.id,
    comment_id: assignment.dataset_comments.id,
    video_title: assignment.dataset_comments.video_title,
    comment: assignment.dataset_comments.comment,
  });
}
