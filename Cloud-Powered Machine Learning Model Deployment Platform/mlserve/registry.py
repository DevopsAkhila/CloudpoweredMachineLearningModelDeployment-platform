import os, json, shutil
from sqlalchemy import select
from .db import SessionLocal, Model, ModelVersion, init_db
from .config import Config

class Registry:
    def __init__(self):
        init_db()
        os.makedirs(Config.ARTIFACT_DIR, exist_ok=True)

    def list_models(self):
        with SessionLocal() as s:
            models = s.query(Model).all()
            out = []
            for m in models:
                versions = [{
                    "version": v.version,
                    "framework": v.framework,
                    "state": v.state,
                    "active": v.active,
                    "created_at": v.created_at.isoformat() if v.created_at else None
                } for v in m.versions]
                out.append({"name": m.name, "versions": versions})
            return out

    def register(self, name, version, framework, file_storage, input_schema=None):
        dest_dir = os.path.join(Config.ARTIFACT_DIR, name, version)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, file_storage.filename)
        file_storage.save(dest_path)

        with SessionLocal() as s:
            model = s.execute(select(Model).where(Model.name == name)).scalar_one_or_none()
            if not model:
                model = Model(name=name)
                s.add(model)
                s.flush()

            # NEW: check if this version already exists
            existing = s.execute(
                select(ModelVersion).where(
                    ModelVersion.model_id == model.id,
                    ModelVersion.version == version
                )
            ).scalar_one_or_none()

            if existing:
                # overwrite fields but keep 'active' flag as-is
                existing.framework = framework
                existing.path = dest_path
                existing.input_schema = input_schema or existing.input_schema
                existing.state = "validated"
                s.commit()
                return {"name": name, "version": version, "framework": framework, "path": dest_path, "updated": True}

            # otherwise create a new row
            mv = ModelVersion(
                model_id=model.id,
                version=version,
                framework=framework,
                path=dest_path,
                input_schema=input_schema or None,
                state="validated",
                active=False
            )
            s.add(mv)
            s.commit()
        return {"name": name, "version": version, "framework": framework, "path": dest_path}
    def activate(self, name, version):
        with SessionLocal() as s:
            model = s.execute(select(Model).where(Model.name == name)).scalar_one_or_none()
            if not model:
                raise ValueError(f"Model '{name}' not found.")
            target = s.execute(select(ModelVersion).where(
                ModelVersion.model_id == model.id, ModelVersion.version == version
            )).scalar_one_or_none()
            if not target:
                raise ValueError(f"Version '{version}' not found for model '{name}'.")
            for mv in model.versions:
                mv.active = (mv.id == target.id)
                if mv.active:
                    mv.state = "deployed"
            s.commit()
        return {"name": name, "active_version": version}
