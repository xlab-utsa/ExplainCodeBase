defmodule Notes do
  def add(list, note), do: [normalize(note) | list]
  defp normalize(note), do: String.trim(note)
end
