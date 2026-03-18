import { supabase } from "./lib/supabase.js";

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "POST only" });
  }

  const {
    token,
    assignment_id,
    comment_id,
    speech_class,
    offensive_class,
    aspects,
  } = req.body;

  if (!token || !assignment_id) {
    return res.status(400).json({ error: "missing fields" });
  }

  // inserir anotação
  const { error: insertError } = await supabase.from("annotations").insert({
    dataset_comment_id: comment_id,
    token,
    speech_class,
    offensive_class,
    aspects,
  });

  if (insertError) {
    return res.status(500).json({ error: insertError.message });
  }

  // marcar assignment como completo
  const { error: updateError } = await supabase
    .from("assignments")
    .update({ completed: true })
    .eq("id", assignment_id);

  if (updateError) {
    return res.status(500).json({ error: updateError.message });
  }

  res.status(200).json({ status: "saved" });
}
