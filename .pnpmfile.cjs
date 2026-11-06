// openapi-typescript (gen:api) drives the TypeScript compiler API, which TypeScript 7 (the native compiler this
// app type-checks with) no longer ships: give it its own TypeScript 5 instead of the app's as its peer.
module.exports = {
  hooks: {
    readPackage(pkg) {
      if (pkg.name === 'openapi-typescript') {
        delete pkg.peerDependencies?.typescript;
        pkg.dependencies = { ...pkg.dependencies, typescript: '^5.9.3' };
      }
      return pkg;
    },
  },
};
