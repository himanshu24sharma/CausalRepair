from models import CausalrepairObservation

class MockAdapter:
    def generate_world(self):
        return {
            "valve": "open",
            "pressure": 50,
            "alarm": False
        }

    def inject_fault(self, world):
        world["valve"] = "closed"

    def render_observation(self, world):
        return CausalrepairObservation(
            description=f"valve={world['valve']}, pressure={world['pressure']}, alarm={world['alarm']}",
            extra=world.copy()
        )

    def diagnose(self, world, entity):
        return f"{entity}={world[entity]}"

    def intervene(self, world, entity, value):
        world[entity] = value

    def propagate(self, world):
        if world["valve"] == "closed":
            world["pressure"] = 90
        else:
            world["pressure"] = 50
        world["alarm"] = world["pressure"] > 80

    def check_constraints(self, world):
        return world["alarm"] == False
