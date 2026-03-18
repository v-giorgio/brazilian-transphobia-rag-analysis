import { supabase } from "./lib/supabase.js";

export default async function handler(req, res) {
  const { token } = req.query;

  const { data } = await supabase
    .from("v_progress")
    .select("*")
    .eq("token", token)
    .single();

  res.status(200).json(data);
}
