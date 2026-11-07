"""r4_loops: bpc sobre TEXTO REAL con la base evaluate() (eval_suite.domain_bpc, 30k posiciones por dominio, mix15 val_*)
para la base y para las recetas que cambian la prediccion tambien con texto real (las demas son identidad sin texto
generado: se verifica con check_equiv y con el mismo bucle).
Uso: python bpc_check.py <pickle_base> <recipe> [<recipe> ...]   (recipe = base | d | a | b | c1 | c2)
Salida: results/bpc_<recipe>.json con bpc, se_block (bloques de 500 posiciones) por dominio."""
import os, sys, json, pickle
os.environ["NUMBA_NUM_THREADS"] = "2"; os.environ["OMP_NUM_THREADS"] = "2"; os.environ["IXCHEL_DATA"] = "mix15"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = __import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "ixchel")); sys.path.insert(0, os.path.join(ROOT, "common"))
import numpy as np
import harness, eval_suite as ES
import gen_lib as G

class Wrap:
    """predict() con la receta; sin texto generado, Pp = pos (identidad para a,b,c1,c2 por construccion)."""
    def __init__(self, m, recipe):
        self.m, self.recipe, self.V = m, recipe, m.V
    def predict(self, ids, pos):
        return G.predict_recipe(self.m, ids, pos, pos, self.recipe, gen_rep=None)

def main():
    pkl = sys.argv[1]; recipes = sys.argv[2:]
    m = pickle.load(open(pkl, "rb"))
    train, val, itos = harness.load(); V = len(itos)
    print("max |predict_recipe(base) - model.predict| =", G.check_equiv(m, val), flush=True)
    for r in recipes:
        model = m if r == "orig" else Wrap(m, r)
        dom = ES.domain_bpc(model, os.path.join(ROOT, "data", "mix15"), V, n_eval=30_000)
        nll = {d: dom[d].pop("nll") for d in dom}
        np.savez(os.path.join(HERE, "results", f"bpc_{r}_nll.npz"), **nll)
        json.dump(dict(recipe=r, domain_bpc=dom), open(os.path.join(HERE, "results", f"bpc_{r}.json"), "w"), indent=1)
        print(r, {d: round(dom[d]["bpc"], 4) for d in dom}, flush=True)

if __name__ == "__main__":
    main()
