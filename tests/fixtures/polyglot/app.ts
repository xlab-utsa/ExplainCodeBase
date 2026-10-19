export class Service {
  constructor(private readonly url: string) {}

  fetch(path: string): string {
    return join(this.url, path);
  }
}

export function join(a: string, b: string): string {
  return `${a}/${b}`;
}

export const main = (): string => new Service("http://x").fetch("y");
